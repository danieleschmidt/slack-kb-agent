import argparse
from pathlib import Path

from .knowledge_base import KnowledgeBase
from .analytics import UsageAnalytics
from .query_processor import QueryProcessor, Query


def main(argv: list[str] | None = None) -> None:
    """Command line interface for querying the knowledge base."""
    parser = argparse.ArgumentParser(description="Query the Slack knowledge base")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--kb", default="kb.json", help="Path to knowledge base JSON")
    parser.add_argument("--analytics", default="analytics.json", help="Path to analytics JSON")
    parser.add_argument("--user", help="User ID")
    parser.add_argument("--channel", help="Channel ID")
    args = parser.parse_args(argv)

    kb = KnowledgeBase.load(Path(args.kb))
    analytics = UsageAnalytics.load(Path(args.analytics))
    processor = QueryProcessor(kb, analytics=analytics)

    docs = processor.process_query(
        Query(text=args.query, user=args.user, channel=args.channel)
    )
    for doc in docs:
        print(doc.content)

    analytics.save(Path(args.analytics))


if __name__ == "__main__":
    main()
