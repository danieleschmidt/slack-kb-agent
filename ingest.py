#!/usr/bin/env python3
"""
Knowledge Source Ingestion Script

This script ingests content from various sources into the knowledge base.

Usage Examples:
    # Ingest local files
    python ingest.py --source files --path ./docs --recursive
    
    # Ingest GitHub repository
    python ingest.py --source github --repo "owner/repo" --token $GITHUB_TOKEN
    
    # Ingest web documentation  
    python ingest.py --source web --url "https://docs.example.com"
    
    # Ingest Slack history
    python ingest.py --source slack --channel "general" --token $SLACK_TOKEN

Required Environment Variables (optional):
    GITHUB_TOKEN    - GitHub personal access token
    SLACK_TOKEN     - Slack bot token (xoxb-...)
"""

import sys
import os
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the main ingestion function
from slack_kb_agent.ingestion import main

if __name__ == "__main__":
    # Set tokens from environment if not provided as arguments
    if "--token" not in sys.argv:
        if "--source" in sys.argv:
            source_idx = sys.argv.index("--source") + 1
            if source_idx < len(sys.argv):
                source_type = sys.argv[source_idx]
                
                if source_type == "github" and os.getenv("GITHUB_TOKEN"):
                    sys.argv.extend(["--token", os.getenv("GITHUB_TOKEN")])
                elif source_type == "slack" and os.getenv("SLACK_TOKEN"):
                    sys.argv.extend(["--token", os.getenv("SLACK_TOKEN")])
    
    exit(main())