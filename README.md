# Slack-KB-Agent

Current version: 1.3.0

Intelligent Slack bot that answers team questions by indexing and searching across documentation, GitHub issues, code comments, and conversation history.

## Features

- **Multi-Source Knowledge Base**: Indexes docs, GitHub issues, code repositories, and Slack history
- **Vector-Based Semantic Search**: Advanced similarity search using sentence transformers and FAISS
  - Understands intent beyond exact keyword matches
  - Configurable similarity thresholds  
  - Hybrid search combining semantic and keyword approaches
  - Automatic fallback to keyword search when vector dependencies unavailable
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
# Index GitHub repositories
python ingest.py --source github --repos "org/repo1,org/repo2"

# Index documentation sites
python ingest.py --source docs --urls "https://docs.example.com"

# Index Slack history (with permissions)
python ingest.py --source slack --channels "general,dev-team"
```

### 6. Production Deployment
```bash
# For production, use process managers like systemd, supervisor, or Docker
# Docker example:
docker-compose up -d
```

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

# Vector Database
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

## Architecture

```
Slack Events → Message Processor → Query Understanding → Knowledge Retrieval → Response Generation → Slack Response
                    ↓                      ↓                    ↓                      ↓
              Intent Classification    Vector Search      Context Assembly    Answer Synthesis
                    ↓                      ↓                    ↓                      ↓
              Knowledge Router        Multiple Sources    Citation Tracking    Quality Check
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
