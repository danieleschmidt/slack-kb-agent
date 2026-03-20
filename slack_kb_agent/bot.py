import os
import re
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from .loader import load_docs
from .search import build_tfidf_index, search, excerpt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCS_DIR = os.environ.get("KB_DOCS_DIR", "./docs")

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
_index = None


def get_index():
    global _index
    if _index is None:
        docs = load_docs(DOCS_DIR)
        if not docs:
            logger.warning(f"No docs found in {DOCS_DIR}")
            _index = {"docs": [], "tfidf": [], "idf": {}}
        else:
            _index = build_tfidf_index(docs)
            logger.info(f"Indexed {len(docs)} docs from {DOCS_DIR}")
    return _index


@app.event("app_mention")
def handle_mention(event, say):
    text = event.get("text", "")
    # Remove the @mention prefix
    query = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    if not query:
        say("Hi! Ask me a question and I'll search our team docs.")
        return

    index = get_index()
    results = search(index, query, top_k=1)

    if not results:
        say(f"No results found for: *{query}*")
        return

    score, doc = results[0]
    snip = excerpt(doc["content"], query)
    filename = doc["filename"]
    say(f"*From `{filename}`:*\n\n{snip}")


def main():
    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    handler.start()


if __name__ == "__main__":
    main()
