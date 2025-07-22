#!/usr/bin/env python3
"""
Slack Knowledge Base Bot Server

This script starts the Slack bot server that can respond to mentions,
direct messages, and slash commands in your Slack workspace.

Usage:
    python bot.py

Required Environment Variables:
    SLACK_BOT_TOKEN     - Bot User OAuth Token (xoxb-...)
    SLACK_APP_TOKEN     - App-Level Token (xapp-...)  
    SLACK_SIGNING_SECRET - Signing Secret for request verification

Optional Environment Variables:
    KB_DATA_PATH        - Path to knowledge base JSON file (default: kb.json)
    ANALYTICS_PATH      - Path to analytics JSON file (default: analytics.json)
    LOG_LEVEL          - Logging level (default: INFO)
"""

import sys
import logging
import os
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from slack_kb_agent import (
    create_bot_from_env, 
    is_slack_bot_available,
    KnowledgeBase,
    UsageAnalytics,
    setup_monitoring,
    MonitoredKnowledgeBase,
    get_global_metrics,
    start_monitoring_server
)


def setup_logging():
    """Configure logging for the bot."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bot.log")
        ]
    )


def load_knowledge_base() -> MonitoredKnowledgeBase:
    """Load knowledge base from file or create empty one with monitoring."""
    kb_path = Path(os.getenv("KB_DATA_PATH", "kb.json"))
    
    if kb_path.exists():
        print(f"Loading knowledge base from {kb_path}")
        kb = KnowledgeBase.load(kb_path)
        print(f"Loaded {len(kb.documents)} documents")
    else:
        print(f"Creating new knowledge base (file not found: {kb_path})")
        kb = KnowledgeBase()
    
    # Wrap with monitoring
    metrics = get_global_metrics()
    monitored_kb = MonitoredKnowledgeBase(kb, metrics)
    
    # Set initial metrics
    metrics.set_gauge("kb_total_documents", len(kb.documents))
    
    return monitored_kb


def load_analytics() -> UsageAnalytics:
    """Load analytics from file or create new tracker."""
    analytics_path = Path(os.getenv("ANALYTICS_PATH", "analytics.json"))
    
    if analytics_path.exists():
        print(f"Loading analytics from {analytics_path}")
        analytics = UsageAnalytics.load(analytics_path)
        print(f"Loaded analytics with {analytics.total_queries} total queries")
    else:
        print(f"Creating new analytics tracker (file not found: {analytics_path})")
        analytics = UsageAnalytics()
    
    return analytics


def check_environment():
    """Check that required environment variables are set."""
    required_vars = [
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN", 
        "SLACK_SIGNING_SECRET"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\nPlease set these variables before running the bot.")
        print("See the README.md for setup instructions.")
        return False
    
    return True


def main():
    """Main entry point for the bot server."""
    print("🤖 Slack Knowledge Base Bot")
    print("=" * 40)
    
    # Set up monitoring first
    monitoring = setup_monitoring()
    if monitoring["status"] == "enabled":
        print("📊 Monitoring system enabled")
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check Slack dependencies
    if not is_slack_bot_available():
        print("❌ Slack bot dependencies not available.")
        print("Install with: pip install slack-bolt slack-sdk")
        raise SystemExit(1)
    
    # Check environment
    if not check_environment():
        raise SystemExit(1)
    
    try:
        # Load knowledge base and analytics
        kb = load_knowledge_base()
        analytics = load_analytics()
        
        # Create bot with loaded data
        bot = create_bot_from_env()
        bot.knowledge_base = kb
        bot.analytics = analytics
        
        print("✅ Bot server configured successfully")
        print(f"📚 Knowledge base: {len(kb.kb.documents)} documents")
        print(f"📊 Vector search: {'enabled' if kb.kb.enable_vector_search else 'disabled'}")
        print(f"📈 Analytics: {analytics.total_queries} total queries")
        print(f"🔍 Monitoring: {'enabled' if monitoring['status'] == 'enabled' else 'disabled'}")
        
        # Start monitoring server if enabled
        if monitoring["status"] == "enabled":
            monitoring_port = int(os.getenv("MONITORING_PORT", "9090"))
            start_monitoring_server(monitoring_port, kb)
            print(f"📊 Monitoring server: http://localhost:{monitoring_port}/health")
            print(f"📈 Metrics endpoint: http://localhost:{monitoring_port}/metrics")
        
        print("\n🚀 Starting bot server...")
        
        # Start the bot
        bot.start()
        
    except KeyboardInterrupt:
        print("\n⏹️  Bot server stopped by user")
        logger.info("Bot server stopped by user interrupt")
    except Exception as e:
        print(f"❌ Error starting bot server: {e}")
        logger.error(f"Failed to start bot server: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()