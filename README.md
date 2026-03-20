# slack-kb-agent

Slack bot that answers team questions by searching your knowledge base docs using TF-IDF keyword search. No LLM required — fully self-contained.

## Features

- Responds to @mentions in Slack
- Loads markdown/text files from a docs directory
- TF-IDF search with zero external ML dependencies
- Returns best matching excerpt from your docs

## Setup

```bash
pip install -r requirements.txt
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_APP_TOKEN=xapp-...
export KB_DOCS_DIR=./docs
python -m slack_kb_agent.bot
```

## Environment Variables

| Variable | Description |
|---|---|
| `SLACK_BOT_TOKEN` | Bot OAuth token (`xoxb-...`) |
| `SLACK_APP_TOKEN` | App-level token for socket mode (`xapp-...`) |
| `KB_DOCS_DIR` | Path to your docs directory (default: `./docs`) |

## Slack App Setup

1. Create a Slack app at https://api.slack.com/apps
2. Enable **Socket Mode** and generate an App-Level Token (`xapp-...`)
3. Add Bot Token Scopes: `app_mentions:read`, `chat:write`
4. Subscribe to `app_mention` events
5. Install app to your workspace

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## How It Works

1. On startup, loads all `.md` and `.txt` files from `KB_DOCS_DIR`
2. Builds a TF-IDF index in memory (pure Python stdlib — no scikit-learn needed)
3. When @mentioned, tokenizes the query and scores each doc
4. Returns the best-matching paragraph excerpt from the top result
