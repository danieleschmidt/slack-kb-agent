#!/usr/bin/env python3
"""
Large Language Model integration for intelligent response generation.

Provides OpenAI/Claude integration for generating contextual responses
based on knowledge base search results.
"""

import os
import logging
import re
import time
from typing import Dict, List, Optional, NamedTuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .models import Document


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMResponse(NamedTuple):
    """Response from LLM generation."""
    content: str
    success: bool
    error_message: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    response_time: Optional[float] = None


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    enabled: bool = True
    provider: str = "openai"  # "openai", "anthropic", "local"
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.1  # Low temperature for factual responses
    max_context_tokens: int = 3000  # Maximum tokens for context
    timeout: int = 30  # API request timeout
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create LLMConfig from environment variables."""
        # Check for API key to determine if LLM should be enabled
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Default to disabled if no API keys are available
        has_api_key = bool(openai_key or anthropic_key)
        enabled = os.getenv("LLM_ENABLED", "true" if has_api_key else "false").lower() == "true"
        
        if enabled and not has_api_key:
            logger.warning("LLM enabled but no API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            enabled = False
        
        # Determine provider and API key
        provider = os.getenv("LLM_PROVIDER", "openai")
        api_key = openai_key if provider == "openai" else anthropic_key
        
        return cls(
            enabled=enabled,
            provider=provider,
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo" if provider == "openai" else "claude-3-haiku-20240307"),
            api_key=api_key,
            api_base=os.getenv("LLM_API_BASE"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_context_tokens=int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "3000")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
            retry_attempts=int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("LLM_RETRY_DELAY", "1.0"))
        )


class PromptTemplate:
    """Template system for LLM prompts."""
    
    def __init__(self, name: str, template: str, required_variables: List[str]):
        self.name = name
        self.template = template
        self.required_variables = required_variables
    
    def format(self, **kwargs) -> str:
        """Format template with provided variables."""
        # Check for required variables
        missing = [var for var in self.required_variables if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        return self.template.format(**kwargs)
    
    @classmethod
    def get_system_prompt(cls) -> 'PromptTemplate':
        """Get system prompt template."""
        return cls(
            name="system_prompt",
            template="""You are {bot_name}, an intelligent knowledge base assistant for a software team.

Your capabilities:
- {capabilities}

Guidelines:
1. Provide accurate, helpful answers based on the provided context
2. If the context doesn't contain relevant information, clearly state that you don't have specific information
3. Keep responses concise but informative
4. Include relevant code examples or commands when helpful
5. If asked about something potentially dangerous, prioritize safety
6. Maintain a professional, helpful tone

Remember: You only know what's in the provided context. Don't make up information.""",
            required_variables=["bot_name", "capabilities"]
        )
    
    @classmethod
    def get_qa_prompt(cls) -> 'PromptTemplate':
        """Get question-answering prompt template."""
        return cls(
            name="qa_prompt",
            template="""Based on the following context from our knowledge base, please answer this question:

Question: {query}

Context:
{context}

Please provide a helpful and accurate answer. If the context doesn't contain enough information to answer the question completely, mention what information is missing.""",
            required_variables=["query", "context"]
        )
    
    @classmethod
    def get_no_context_prompt(cls) -> 'PromptTemplate':
        """Get prompt for when no context is available."""
        return cls(
            name="no_context_prompt",
            template="""I was asked: "{query}"

Unfortunately, I don't have specific information about this topic in my knowledge base. Here are some suggestions:

1. Try rephrasing your question with different keywords
2. Check if this is covered in our documentation
3. Ask a team member who might have experience with this topic

Is there anything else I can help you with?""",
            required_variables=["query"]
        )


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response from the LLM."""
        pass
    
    @staticmethod
    def create(config: LLMConfig) -> 'LLMProvider':
        """Factory method to create appropriate provider."""
        if config.provider == "openai":
            return OpenAIProvider(config)
        elif config.provider == "anthropic":
            return AnthropicProvider(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._import_openai()
    
    def _import_openai(self):
        """Import OpenAI with proper error handling."""
        try:
            import openai
            self.openai = openai
            
            # Configure API key and base
            if self.config.api_key:
                openai.api_key = self.config.api_key
            if self.config.api_base:
                openai.api_base = self.config.api_base
                
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install with: pip install openai"
            )
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai.ChatCompletion.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            
            content = response.choices[0].message.content
            usage = response.get('usage', {})
            
            return LLMResponse(
                content=content,
                success=True,
                token_usage={
                    'prompt': usage.get('prompt_tokens', 0),
                    'completion': usage.get('completion_tokens', 0),
                    'total': usage.get('total_tokens', 0)
                },
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"OpenAI API error: {e}")
            
            return LLMResponse(
                content="",
                success=False,
                error_message=str(e),
                response_time=response_time
            )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._import_anthropic()
    
    def _import_anthropic(self):
        """Import Anthropic with proper error handling."""
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=self.config.api_key)
        except ImportError:
            raise ImportError(
                "Anthropic package not found. Install with: pip install anthropic"
            )
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using Anthropic Claude API."""
        start_time = time.time()
        
        try:
            # Combine system prompt with user prompt for Claude
            full_prompt = ""
            if system_prompt:
                full_prompt += f"System: {system_prompt}\n\n"
            full_prompt += f"Human: {prompt}\n\nAssistant:"
            
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            response_time = time.time() - start_time
            
            content = response.content[0].text
            usage = response.usage
            
            return LLMResponse(
                content=content,
                success=True,
                token_usage={
                    'prompt': usage.input_tokens,
                    'completion': usage.output_tokens,
                    'total': usage.input_tokens + usage.output_tokens
                },
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Anthropic API error: {e}")
            
            return LLMResponse(
                content="",
                success=False,
                error_message=str(e),
                response_time=response_time
            )


class ResponseGenerator:
    """Main class for generating intelligent responses."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.provider = None
        
        if self.config.enabled:
            try:
                self.provider = LLMProvider.create(self.config)
                logger.info(f"LLM integration enabled with {self.config.provider} ({self.config.model})")
            except Exception as e:
                logger.error(f"Failed to initialize LLM provider: {e}")
                self.config.enabled = False
        else:
            logger.info("LLM integration disabled")
    
    def generate_response(self, 
                         query: str, 
                         context_documents: List[Document],
                         user_id: Optional[str] = None) -> LLMResponse:
        """Generate intelligent response based on query and context."""
        
        if not self.config.enabled or not self.provider:
            return LLMResponse(
                content="",
                success=False,
                error_message="LLM integration is disabled or not available"
            )
        
        try:
            # Sanitize query to prevent prompt injection
            sanitized_query = self._sanitize_query(query)
            
            # Prepare context from documents
            context = self._prepare_context(context_documents)
            
            # Choose appropriate prompt template
            if context.strip():
                template = PromptTemplate.get_qa_prompt()
                prompt = template.format(query=sanitized_query, context=context)
            else:
                template = PromptTemplate.get_no_context_prompt()
                prompt = template.format(query=sanitized_query)
            
            # Generate system prompt
            system_template = PromptTemplate.get_system_prompt()
            system_prompt = system_template.format(
                bot_name="Slack KB Agent",
                capabilities="search knowledge base, answer technical questions, provide code examples"
            )
            
            # Generate response with retry logic
            for attempt in range(self.config.retry_attempts):
                try:
                    response = self.provider.generate_response(prompt, system_prompt)
                    
                    if response.success:
                        # Log successful generation
                        logger.info(f"Generated response for user {user_id}: {len(response.content)} chars, "
                                  f"{response.token_usage.get('total', 0) if response.token_usage else 0} tokens")
                        return response
                    
                    # Log failed attempt
                    logger.warning(f"LLM generation attempt {attempt + 1} failed: {response.error_message}")
                    
                    # If this was the last attempt, return the failed response
                    if attempt == self.config.retry_attempts - 1:
                        return response
                    
                    # Wait before retry
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    
                except Exception as e:
                    logger.error(f"LLM generation error on attempt {attempt + 1}: {e}")
                    if attempt == self.config.retry_attempts - 1:
                        return LLMResponse(
                            content="",
                            success=False,
                            error_message=f"Failed to generate response after {self.config.retry_attempts} attempts: {e}"
                        )
            
        except Exception as e:
            logger.error(f"Unexpected error in response generation: {e}")
            return LLMResponse(
                content="",
                success=False,
                error_message=f"Unexpected error: {e}"
            )
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize query to prevent prompt injection attacks."""
        # Remove potentially dangerous instruction patterns
        dangerous_patterns = [
            r"ignore\s+(?:previous\s+)?instructions",
            r"system\s*:",
            r"new\s+instructions?\s*:",
            r"override\s+",
            r"jailbreak",
            r"prompt\s+injection"
        ]
        
        sanitized = query
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        # Limit length to prevent token overflow
        max_query_length = 500
        if len(sanitized) > max_query_length:
            sanitized = sanitized[:max_query_length] + "..."
        
        return sanitized.strip()
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context string from documents with token management."""
        if not documents:
            return ""
        
        context_parts = []
        total_tokens = 0
        max_tokens = self.config.max_context_tokens
        
        # Rough token estimation (4 chars per token)
        chars_per_token = 4
        
        for i, doc in enumerate(documents):
            # Format document with source information
            doc_text = f"Source: {doc.source}\nContent: {doc.content}"
            
            # Estimate tokens for this document
            doc_tokens = len(doc_text) // chars_per_token
            
            # Check if adding this document would exceed token limit
            if total_tokens + doc_tokens > max_tokens:
                # Try to include partial document if space allows
                remaining_chars = (max_tokens - total_tokens) * chars_per_token
                if remaining_chars > 100:  # Only include if meaningful amount
                    partial_content = doc.content[:remaining_chars - 50]  # Leave room for source info
                    doc_text = f"Source: {doc.source}\nContent: {partial_content}..."
                    context_parts.append(doc_text)
                
                logger.debug(f"Context truncated at {i+1}/{len(documents)} documents due to token limit")
                break
            
            context_parts.append(doc_text)
            total_tokens += doc_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    def is_available(self) -> bool:
        """Check if LLM integration is available."""
        return self.config.enabled and self.provider is not None


# Global response generator instance
_global_response_generator: Optional[ResponseGenerator] = None


def get_response_generator() -> ResponseGenerator:
    """Get global response generator instance."""
    global _global_response_generator
    
    if _global_response_generator is None:
        config = LLMConfig.from_env()
        _global_response_generator = ResponseGenerator(config)
    
    return _global_response_generator


def generate_intelligent_response(query: str, 
                                context_documents: List[Document],
                                user_id: Optional[str] = None) -> LLMResponse:
    """Convenience function for generating intelligent responses."""
    generator = get_response_generator()
    return generator.generate_response(query, context_documents, user_id)