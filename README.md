# Slack-KB-Agent

Current version: 1.7.2

Intelligent Slack bot that answers team questions by indexing and searching across documentation, GitHub issues, code comments, and conversation history.

## Features

- **Multi-Source Knowledge Base**: Indexes docs, GitHub issues, code repositories, and Slack history
- **Advanced Search Engine**: High-performance search with multiple approaches
  - **Indexed Search**: Inverted index with TF-IDF scoring for fast keyword search
  - **Vector-Based Semantic Search**: Advanced similarity search using sentence transformers and FAISS
  - **Hybrid Search**: Combines semantic and keyword approaches with configurable weights
  - Understands intent beyond exact keyword matches with configurable similarity thresholds
  - Automatic fallback to optimized keyword search when vector dependencies unavailable
- **Contextual Q&A**: Understands team-specific terminology and project context
- **Real-time Learning**: Continuously updates knowledge base from ongoing conversations
- **Smart Routing**: Escalates complex questions to appropriate team members
- **Usage Analytics**: Tracks common questions, knowledge gaps, active users, and active channels
  (powered by the `UsageAnalytics` module integrated with the `QueryProcessor`)
- **Analytics Persistence**: Save and load usage statistics from JSON files
- **Knowledge Base Persistence**: Save and load indexed documents from JSON files
- **Command Line Interface**: Query the knowledge base using the `slack-kb-agent` CLI
- **Permission-Aware**: Respects access controls and sensitive information boundaries

## Quick Setup

### 1. Install Dependencies
```bash
# Install core dependencies
pip install -e .

# For full functionality, install optional dependencies
pip install sentence-transformers faiss-cpu torch slack-bolt slack-sdk
```

### 2. Create Slack App
1. Go to [Slack API](https://api.slack.com/apps) and create a new app
2. Enable Socket Mode in your app settings
3. Add the following OAuth scopes under "OAuth & Permissions":
   - `chat:write` - Send messages
   - `app_mentions:read` - Receive @mentions
   - `im:read` - Read direct messages
   - `channels:read` - Access channel information
4. Install the app to your workspace

### 3. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your Slack credentials:
# - SLACK_BOT_TOKEN (from OAuth & Permissions)
# - SLACK_APP_TOKEN (from Basic Information > App-Level Tokens)
# - SLACK_SIGNING_SECRET (from Basic Information)
```

### 4. Start the Bot
```bash
# Start the Slack bot server
python bot.py
```

The bot will connect to Slack and respond to:
- **@mentions**: `@kb-agent How do I deploy the application?`
- **Direct messages**: Send a DM with your question
- **Slash commands**: `/kb search query` or `/kb help`

### 5. Knowledge Source Integration
```bash
# Index local files and documentation
python ingest.py --source files --path ./docs --recursive

# Index GitHub repositories  
python ingest.py --source github --repo "owner/repo" --token $GITHUB_TOKEN

# Index web documentation sites
python ingest.py --source web --url "https://docs.example.com"

# Index Slack conversation history
python ingest.py --source slack --channel "general" --token $SLACK_BOT_TOKEN --days 30
```

The ingested content is automatically available to the bot on next restart.

### 6. Production Deployment
```bash
# Start the bot server
python bot.py

# Start monitoring server (separate process)
python monitoring_server.py

# For production, use process managers like systemd, supervisor, or Docker
# Docker example:
docker-compose up -d
```

### 7. Monitoring & Observability
The system includes comprehensive monitoring for production deployments:

```bash
# Access monitoring endpoints
curl http://localhost:9090/health     # Health check
curl http://localhost:9090/metrics    # Prometheus metrics  
curl http://localhost:9090/status     # Service status
```

Available metrics:
- **Request metrics**: `slack_messages_received`, `slack_responses_sent`
- **Performance**: `query_duration_seconds`, `search_response_time`
- **Knowledge base**: `kb_total_documents`, `kb_search_queries`
- **System health**: Memory usage, disk space, service status

## Usage

### In Slack Channels
```
@kb-agent How do I set up the development environment?
@kb-agent What was the resolution to issue #1234?
@kb-agent Where is the API documentation for the auth service?
```

### Direct Messages
```
How do I deploy the application?
What are the authentication options?
Show me the troubleshooting guide
```

### Slash Commands
- `/kb <query>` - Search knowledge base: `/kb deployment process`
- `/kb help` - Show help and usage instructions
- `/kb stats` - Show usage statistics and analytics
- See [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md) for programmatic examples

### CLI
```bash
$ slack-kb-agent "deployment process" --kb kb.json
```

## Configuration

### Environment Variables
```env
# Slack Configuration
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_APP_TOKEN=xapp-your-app-token

# Knowledge Sources
GITHUB_TOKEN=ghp_your-github-token
GITHUB_ORGS=your-org1,your-org2

# PostgreSQL Database (for persistent storage)
DATABASE_URL=postgresql://username:password@localhost:5432/slack_kb_agent

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# Vector Database (optional)
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-east1-gcp

# LLM Configuration
OPENAI_API_KEY=sk-your-openai-key
MODEL_NAME=gpt-4
TEMPERATURE=0.1
```

### Bot Configuration
```yaml
# config/bot_config.yml
knowledge_sources:
  github:
    enabled: true
    repositories: ["org/repo1", "org/repo2"]
    include_issues: true
    include_prs: true
    include_code_comments: true
  
  documentation:
    enabled: true
    sites:
      - url: "https://docs.company.com"
        crawl_depth: 3
      - url: "https://wiki.company.com"
        auth_required: true
  
  slack:
    enabled: true
    channels: ["general", "dev-team", "support"]
    history_days: 90
    exclude_private: true

response_config:
  max_context_length: 4000
  include_sources: true
  confidence_threshold: 0.7
  fallback_to_human: true

permissions:
  admin_users: ["user1", "user2"]
  restricted_channels: ["hr-private", "finance"]
  public_channels: ["general", "random"]
```

## Database & Persistence

The system supports both PostgreSQL database persistence and JSON file storage:

### PostgreSQL Database
- **Production-ready**: ACID compliance, concurrent access, connection pooling
- **Scalable**: Handle large knowledge bases with efficient indexing
- **Backup/Restore**: Built-in backup and restore functionality
- **Migration**: Alembic-based schema migrations

### Database Management
```bash
# Initialize database schema
slack-kb-db init

# Check database status
slack-kb-db check

# Create backup
slack-kb-db backup /path/to/backup.json.gz

# Restore from backup
slack-kb-db restore /path/to/backup.json.gz --clear

# Migrate JSON file to database
slack-kb-db migrate /path/to/knowledge_base.json

# Export database to JSON
slack-kb-db export /path/to/export.json

# Validate backup file
slack-kb-db validate /path/to/backup.json.gz
```

### Environment Setup
```bash
# PostgreSQL (required for database persistence)
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres createdb slack_kb_agent

# Set database URL
export DATABASE_URL="postgresql://username:password@localhost:5432/slack_kb_agent"
```

## Architecture

```
Slack Events → Message Processor → Query Understanding → Knowledge Retrieval → Response Generation → Slack Response
                    ↓                      ↓                    ↓                      ↓
              Intent Classification    Vector Search      Context Assembly    Answer Synthesis
                    ↓                      ↓                    ↓                      ↓
              Knowledge Router        Multiple Sources    Citation Tracking    Quality Check
                                           ↓
                                  PostgreSQL Database
                                  (Persistent Storage)
```

## Knowledge Sources

### GitHub Integration
- **Issues & PRs**: Search through titles, descriptions, and comments
- **Code Comments**: Index inline documentation and comments
- **Wiki Pages**: Include repository wikis and README files
- **Releases**: Track feature releases and changelog information

### Documentation Sites
- **Web Crawling**: Automatically crawl and index documentation sites
- **Markdown Processing**: Parse and structure markdown documentation
- **API Docs**: Special handling for OpenAPI/Swagger specifications
- **Change Detection**: Monitor for documentation updates

### Slack History
- **Message Indexing**: Search through historical conversations
- **Thread Context**: Maintain conversation context and replies
- **File Attachments**: Index shared documents and images (OCR)
- **Reaction Analysis**: Identify helpful responses based on reactions

## Knowledge Ingestion System

The ingestion system automatically processes and filters content from multiple sources:

### Automatic Content Processing
```python
from slack_kb_agent import FileIngester, GitHubIngester, WebDocumentationCrawler

# File-based ingestion with automatic format detection
ingester = FileIngester()
documents = ingester.ingest_directory("./docs", recursive=True)

# GitHub integration with issues and README
github = GitHubIngester(token="your-token")
documents = github.ingest_repository("owner/repo", include_issues=True)

# Web documentation crawling
crawler = WebDocumentationCrawler()
documents = crawler.crawl_url("https://docs.example.com", max_depth=2)
```

### Security Features
- **Sensitive Data Detection**: Automatically detects and redacts API keys, passwords, tokens
- **Content Filtering**: Removes low-value content and excessive whitespace
- **Incremental Updates**: Avoids duplicate ingestion with checksum tracking

### Supported Formats
- **Text Files**: `.txt`, `.md`, `.rst` with automatic markdown processing
- **Code Files**: `.py`, `.js`, `.ts`, `.json`, `.yaml` with syntax preservation  
- **Web Content**: HTML with automatic content extraction and link following
- **GitHub**: Issues, pull requests, README files, and repository metadata
- **Slack**: Message history with user attribution and threading context

## Advanced Features

### Smart Query Processing
```python
# Query understanding and intent classification
class QueryProcessor:
    def process_query(self, message):
        intent = self.classify_intent(message)
        entities = self.extract_entities(message)
        context = self.get_conversation_context(message)
        return self.route_query(intent, entities, context)
```

### Vector-Based Semantic Search
```python
from slack_kb_agent import KnowledgeBase, Document

# Create knowledge base with vector search enabled
kb = KnowledgeBase(
    enable_vector_search=True,
    vector_model="all-MiniLM-L6-v2",  # Sentence transformer model
    similarity_threshold=0.5          # Minimum similarity score
)

# Add documents
kb.add_document(Document(content="Python programming tutorial", source="docs"))
kb.add_document(Document(content="Machine learning with PyTorch", source="docs"))

# Semantic search - finds related content beyond exact keywords
results = kb.search_semantic("coding in Python", threshold=0.7)

# Hybrid search - combines semantic and keyword matching
results = kb.search_hybrid("Python development", vector_weight=0.7, keyword_weight=0.3)

# Traditional keyword search still available
results = kb.search("Python")
```

### Contextual Responses
- **Team Terminology**: Learns team-specific jargon and abbreviations
- **Project Context**: Understands current project states and priorities
- **Historical Awareness**: References past decisions and discussions
- **User Personalization**: Adapts responses based on user expertise level

### Analytics Dashboard
- **Popular Questions**: Track most frequently asked questions
- **Knowledge Gaps**: Identify areas lacking documentation
- **Response Quality**: Monitor answer accuracy and user satisfaction
- **Usage Patterns**: Understand team information-seeking behavior

## Sample Interactions

### Technical Question
```
User: @kb-agent How do I authenticate with the API?

Bot: Based on our API documentation, here are the authentication methods:

1. **API Key Authentication** (Recommended)
   - Add header: `Authorization: Bearer YOUR_API_KEY`
   - Get your key from the dashboard: https://app.company.com/keys

2. **OAuth 2.0** (For user-specific actions)
   - Follow the OAuth flow documented here: [docs link]

📚 Sources: API Documentation, GitHub issue #456
💡 Pro tip: From @john.doe's comment in #dev-team: "Always use environment variables for API keys"
```

### Project Status Question
```
User: What's the status of the payment integration?

Bot: Based on recent activity:

🚧 **Current Status**: In progress
📅 **Last Update**: 2 days ago by @sarah.dev
🎯 **Progress**: 3/5 tasks completed

**Recent Updates**:
- ✅ Payment gateway integration (completed)
- ✅ Basic error handling (completed) 
- ✅ Unit tests (completed)
- 🔄 Webhook processing (in progress)
- ⏳ Integration testing (pending)

📚 Sources: GitHub PR #123, Slack #dev-team discussion, Project board
```

## Deployment Options

### Self-Hosted
```bash
# Docker deployment
docker-compose up -d

# Kubernetes
kubectl apply -f k8s/

# Traditional server
python bot.py --port 3000
```

### Cloud Platforms
- **Heroku**: One-click deployment with add-ons
- **AWS Lambda**: Serverless deployment for cost efficiency
- **Google Cloud Run**: Containerized deployment with auto-scaling
- **Azure Container Instances**: Simple container deployment

## Security & Privacy

### Data Protection
- **Encryption**: All data encrypted at rest and in transit
- **Access Controls**: Respects Slack channel permissions
- **Data Retention**: Configurable retention policies
- **Audit Logging**: Complete audit trail of all queries and responses

### Privacy Features
- **Sensitive Data Detection**: Automatically redacts PII and secrets
- **Permission Boundaries**: Only accesses authorized channels and repositories
- **User Consent**: Opt-in for personal message indexing
- **Data Deletion**: Support for right-to-be-forgotten requests

## Contributing

We welcome contributions in these areas:
- Additional knowledge source integrations
- Enhanced natural language processing
- UI/UX improvements for Slack interactions
- Performance optimizations
- Security enhancements

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://docs.slack-kb-agent.com)
- 💬 [Community Discord](https://discord.gg/slack-kb-agent)
- 🐛 [Issue Tracker](https://github.com/your-org/slack-kb-agent/issues)
- 📧 Email: support@slack-kb-agent.com
