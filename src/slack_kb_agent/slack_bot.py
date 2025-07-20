"""Slack bot server for handling real-time events and interactions."""

from __future__ import annotations

import logging
import os
import re
from typing import List, Dict, Any, Optional

from .knowledge_base import KnowledgeBase
from .models import Document
from .analytics import UsageAnalytics
from .query_processor import QueryProcessor
from .validation import validate_slack_input, sanitize_query, get_validator
from .rate_limiting import get_user_rate_limiter, RateLimitResult
from .llm import get_response_generator, LLMResponse

logger = logging.getLogger(__name__)

# Optional Slack dependencies with graceful fallback
try:
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    from slack_sdk import WebClient
    SLACK_DEPS_AVAILABLE = True
except ImportError:
    SLACK_DEPS_AVAILABLE = False
    App = None
    SocketModeHandler = None
    WebClient = None


class SlackBotServer:
    """Slack bot server for handling knowledge base queries."""
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        slack_bot_token: str,
        slack_app_token: str,
        signing_secret: str,
        *,
        analytics: Optional[UsageAnalytics] = None,
        max_results: int = 5,
        response_timeout: int = 30
    ):
        """Initialize Slack bot server.
        
        Args:
            knowledge_base: KnowledgeBase instance for querying
            slack_bot_token: Bot User OAuth Token (xoxb-...)
            slack_app_token: App-Level Token (xapp-...)
            signing_secret: Slack signing secret for request verification
            analytics: Optional analytics tracker
            max_results: Maximum number of results to return
            response_timeout: Response timeout in seconds
        """
        if not SLACK_DEPS_AVAILABLE:
            raise ImportError(
                "Slack dependencies not available. "
                "Install with: pip install slack-bolt slack-sdk"
            )
        
        # Validate tokens (basic security check)
        if not slack_bot_token.startswith("xoxb-"):
            raise ValueError("Invalid bot token format. Must start with 'xoxb-'")
        if not slack_app_token.startswith("xapp-"):
            raise ValueError("Invalid app token format. Must start with 'xapp-'")
        if len(signing_secret) < 10:
            raise ValueError("Signing secret too short. Must be at least 10 characters")
        
        self.knowledge_base = knowledge_base
        self.slack_bot_token = slack_bot_token
        self.slack_app_token = slack_app_token
        self.signing_secret = signing_secret
        self.analytics = analytics
        self.max_results = max_results
        self.response_timeout = response_timeout
        
        # Initialize query processor
        self.query_processor = QueryProcessor(
            kb=knowledge_base,
            analytics=analytics
        )
        
        # Initialize Slack app
        self.app = App(
            token=slack_bot_token,
            signing_secret=signing_secret
        )
        
        # Register event handlers
        self._register_handlers()
        
        logger.info("Slack bot server initialized")
    
    def _register_handlers(self) -> None:
        """Register Slack event handlers."""
        
        @self.app.event("app_mention")
        def handle_app_mention(event, say, client):
            """Handle @bot mentions in channels."""
            try:
                self._handle_query_event(event, say, client, is_mention=True)
            except Exception as e:
                logger.error(f"Error handling app mention: {e}")
                say("Sorry, I encountered an error processing your request.")
        
        @self.app.event("message")
        def handle_direct_message(event, say, client):
            """Handle direct messages to the bot."""
            # Only respond to DMs (channel_type is 'im')
            if event.get("channel_type") == "im":
                try:
                    self._handle_query_event(event, say, client, is_mention=False)
                except Exception as e:
                    logger.error(f"Error handling direct message: {e}")
                    say("Sorry, I encountered an error processing your request.")
        
        @self.app.command("/kb")
        def handle_kb_command(ack, command, say, client):
            """Handle /kb slash commands."""
            ack()  # Acknowledge the command immediately
            try:
                query = command.get("text", "").strip()
                user_id = command.get("user_id")
                channel_id = command.get("channel_id")
                
                # Check rate limiting for slash commands
                rate_limiter = get_user_rate_limiter()
                rate_result = rate_limiter.check_user_rate_limit(user_id, query)
                if not rate_result.allowed:
                    logger.warning(f"Slash command rate limit exceeded for user {user_id}: {rate_result.error_message}")
                    say(f"⏰ {rate_result.error_message}")
                    return
                
                if not query:
                    say(self._get_help_text())
                    return
                
                # Process special commands (these are safe)
                if query.lower() in ["help", "--help", "-h"]:
                    say(self._get_help_text())
                    return
                elif query.lower() in ["stats", "statistics"]:
                    say(self._get_stats_text())
                    return
                
                # Sanitize and validate the query for search
                sanitized_query = sanitize_query(query)
                if not sanitized_query:
                    logger.warning(f"Dangerous slash command query blocked from user {user_id}: {query}")
                    say("Sorry, I can't process that request. Please try a different search query.")
                    return
                
                # Log if query was modified during sanitization
                if sanitized_query != query:
                    logger.info(f"Slash command query sanitized for user {user_id}: '{query}' -> '{sanitized_query}'")
                
                # Process search query
                results = self.process_query(sanitized_query, user_id, channel_id)
                response = self.format_response(results, sanitized_query, user_id)
                say(response)
                
            except Exception as e:
                logger.error(f"Error handling slash command: {e}")
                say("Sorry, I encountered an error processing your command.")
    
    def _handle_query_event(self, event: Dict[str, Any], say, client, is_mention: bool) -> None:
        """Handle query events from mentions or DMs."""
        user_id = event.get("user")
        
        # Check rate limiting first
        rate_limiter = get_user_rate_limiter()
        rate_result = rate_limiter.check_user_rate_limit(user_id, event.get("text", ""))
        if not rate_result.allowed:
            logger.warning(f"Rate limit exceeded for user {user_id}: {rate_result.error_message}")
            say(f"⏰ {rate_result.error_message}")
            return
        
        # Validate Slack input
        validation_result = validate_slack_input(event)
        if not validation_result.is_valid:
            logger.warning(f"Invalid Slack input from user {user_id}: {validation_result.error_message}")
            say("Sorry, I couldn't process your request. Please try rephrasing your question.")
            return
        
        text = event.get("text", "")
        channel_id = event.get("channel")
        
        # Extract query from mention (remove bot mention)
        if is_mention:
            # Remove <@UBOT_ID> from the text
            bot_mention_pattern = r'<@[UW][A-Z0-9]+>'
            query = re.sub(bot_mention_pattern, '', text).strip()
        else:
            query = text.strip()
        
        if not query:
            say(self._get_help_text())
            return
        
        # Sanitize and validate the query
        sanitized_query = sanitize_query(query)
        if not sanitized_query:
            logger.warning(f"Dangerous query blocked from user {user_id}: {query}")
            say("Sorry, I can't process that request. Please try a different question.")
            return
        
        # Log if query was modified during sanitization
        if sanitized_query != query:
            logger.info(f"Query sanitized for user {user_id}: '{query}' -> '{sanitized_query}'")
        
        # Process the sanitized query
        results = self.process_query(sanitized_query, user_id, channel_id)
        response = self.format_response(results, sanitized_query, user_id)
        say(response)
    
    def process_query(
        self, 
        query: str, 
        user_id: Optional[str] = None, 
        channel_id: Optional[str] = None
    ) -> List[Document]:
        """Process a user query and return relevant documents.
        
        Args:
            query: User's search query
            user_id: Slack user ID (for analytics)
            channel_id: Slack channel ID (for analytics)
            
        Returns:
            List of relevant documents
        """
        if not query.strip():
            return []
        
        try:
            # Use semantic search if available, fallback to keyword search
            if hasattr(self.knowledge_base, 'search_semantic'):
                results = self.knowledge_base.search_semantic(query, top_k=self.max_results)
            else:
                results = self.knowledge_base.search(query)[:self.max_results]
            
            # Record analytics if available
            if self.analytics and user_id and channel_id:
                self.analytics.record_query(query, user=user_id, channel=channel_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            # Fallback to basic search on error
            try:
                return self.knowledge_base.search(query)[:self.max_results]
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []
    
    def format_response(self, documents: List[Document], query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Format search results for Slack response with optional LLM enhancement.
        
        Args:
            documents: List of relevant documents
            query: Original search query
            user_id: User ID for LLM personalization
            
        Returns:
            Slack response payload
        """
        if not query.strip():
            return {
                "text": self._get_help_text(),
                "response_type": "ephemeral"
            }
        
        # Try to generate intelligent response using LLM
        response_generator = get_response_generator()
        if response_generator.is_available() and documents:
            try:
                llm_response = response_generator.generate_response(
                    query=query,
                    context_documents=documents,
                    user_id=user_id
                )
                
                if llm_response.success and llm_response.content.strip():
                    # Use LLM-generated response with source citations
                    response_text = f"🤖 {llm_response.content}\n\n"
                    
                    # Add compact source citations
                    if len(documents) <= 3:
                        response_text += "📚 **Sources:** "
                        sources = [f"{doc.source}" for doc in documents[:3]]
                        response_text += ", ".join(sources)
                    else:
                        response_text += f"📚 **Based on {len(documents)} sources** including: "
                        sources = [f"{doc.source}" for doc in documents[:2]]
                        response_text += ", ".join(sources) + f" and {len(documents)-2} more"
                    
                    response_text += "\n\n💡 _Ask follow-up questions or use `/kb help` for more options._"
                    
                    return {
                        "text": response_text,
                        "response_type": "in_channel"
                    }
                    
            except Exception as e:
                logger.warning(f"LLM response generation failed, falling back to basic format: {e}")
        
        # Fallback to traditional document listing format
        if not documents:
            return {
                "text": f"🔍 No results found for '{query}'. Try rephrasing your question or check the available documentation.",
                "response_type": "ephemeral"
            }
        
        # Build traditional response with formatted results
        response_text = f"📚 Found {len(documents)} result{'s' if len(documents) != 1 else ''} for '{query}':\n\n"
        
        for i, doc in enumerate(documents, 1):
            # Truncate content for readability
            content = doc.content
            if len(content) > 200:
                content = content[:197] + "..."
            
            response_text += f"*{i}. From {doc.source}*\n"
            response_text += f"```{content}```\n"
            
            # Add metadata if available
            if doc.metadata:
                if "path" in doc.metadata:
                    response_text += f"📁 _{doc.metadata['path']}_\n"
                elif "user" in doc.metadata:
                    response_text += f"👤 _{doc.metadata['user']}_\n"
            
            response_text += "\n"
        
        # Add help footer
        response_text += "💡 _Use `/kb help` for more options or ask follow-up questions._"
        
        return {
            "text": response_text,
            "response_type": "in_channel"
        }
    
    def _get_help_text(self) -> str:
        """Generate help text for users."""
        return """🤖 *Slack Knowledge Base Assistant*

*How to use:*
• @mention me with your question: `@kb-agent How do I deploy the app?`
• Use slash commands: `/kb deployment process`
• Send me a direct message

*Available commands:*
• `/kb help` - Show this help
• `/kb stats` - Show usage statistics
• `/kb <your question>` - Search the knowledge base

*Examples:*
• `@kb-agent API authentication`
• `/kb troubleshooting guide`
• `How do I set up the development environment?`

I can search through documentation, code comments, GitHub issues, and team knowledge to help answer your questions!"""
    
    def _get_stats_text(self) -> str:
        """Generate statistics text."""
        if not self.analytics:
            return "📊 Analytics not enabled."
        
        stats = f"""📊 *Knowledge Base Statistics*

• Total queries: {self.analytics.total_queries}
• Total documents: {len(self.knowledge_base.documents)}
• Vector search: {'✅ Enabled' if self.knowledge_base.enable_vector_search else '❌ Disabled'}

*Top queries:*
"""
        
        for i, (query, count) in enumerate(self.analytics.top_queries(5), 1):
            stats += f"{i}. _{query}_ ({count} times)\n"
        
        return stats
    
    def start(self) -> None:
        """Start the bot server using Socket Mode."""
        if not SLACK_DEPS_AVAILABLE:
            raise ImportError("Slack dependencies not available")
        
        try:
            handler = SocketModeHandler(self.app, self.slack_app_token)
            logger.info("Starting Slack bot server...")
            handler.start()
        except Exception as e:
            logger.error(f"Failed to start bot server: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the bot server gracefully."""
        # The SocketModeHandler doesn't have a built-in stop method
        # This is a placeholder for future implementation
        logger.info("Bot server stopped")


def create_bot_from_env() -> SlackBotServer:
    """Create bot server from environment variables.
    
    Expected environment variables:
    - SLACK_BOT_TOKEN: Bot User OAuth Token
    - SLACK_APP_TOKEN: App-Level Token  
    - SLACK_SIGNING_SECRET: Signing Secret
    
    Returns:
        Configured SlackBotServer instance
    """
    required_env_vars = {
        "SLACK_BOT_TOKEN": "Bot User OAuth Token",
        "SLACK_APP_TOKEN": "App-Level Token",
        "SLACK_SIGNING_SECRET": "Signing Secret"
    }
    
    missing_vars = []
    for var, description in required_env_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables:\n" +
            "\n".join(f"- {var}" for var in missing_vars)
        )
    
    # Create knowledge base and analytics
    kb = KnowledgeBase()
    analytics = UsageAnalytics()
    
    return SlackBotServer(
        knowledge_base=kb,
        slack_bot_token=os.getenv("SLACK_BOT_TOKEN"),
        slack_app_token=os.getenv("SLACK_APP_TOKEN"),
        signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
        analytics=analytics
    )


def is_slack_bot_available() -> bool:
    """Check if Slack bot dependencies are available."""
    return SLACK_DEPS_AVAILABLE