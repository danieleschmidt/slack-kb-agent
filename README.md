# Slack-KB-Agent

Intelligent Slack bot that answers team questions by indexing and searching across documentation, GitHub issues, code comments, and conversation history.

## Features

- **Multi-Source Knowledge Base**: Indexes docs, GitHub issues, code repositories, and Slack history
- **Contextual Q&A**: Understands team-specific terminology and project context
- **Real-time Learning**: Continuously updates knowledge base from ongoing conversations
- **Smart Routing**: Escalates complex questions to appropriate team members
- **Usage Analytics**: Tracks common questions and knowledge gaps
- **Permission-Aware**: Respects access controls and sensitive information boundaries

## Quick Setup

### 1. Slack App Configuration
```bash
# Install dependencies
npm install
# or
pip install -r requirements.txt

# Set up Slack app credentials
cp .env.example .env
# Edit .env with your Slack tokens
```

### 2. Knowledge Source Integration
```bash
# Index GitHub repositories
python ingest.py --source github --repos "org/repo1,org/repo2"

# Index documentation sites
python ingest.py --source docs --urls "https://docs.example.com"

# Index Slack history (with permissions)
python ingest.py --source slack --channels "general,dev-team"
```

### 3. Deploy Bot
```bash
# Start the bot
python bot.py

# Or use Docker
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
/kb search deployment process
/kb recent updates on project alpha
/kb who worked on the billing module last?
```

### Slash Commands
- `/kb search <query>` - Search knowledge base
- `/kb add <url>` - Add documentation URL to index
- `/kb stats` - Show usage statistics
- `/kb feedback <rating>` - Rate last response

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
