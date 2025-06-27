# API Usage Guide

This guide provides examples for interacting with the `slack_kb_agent` package programmatically.

## Querying the Knowledge Base

```python
from slack_kb_agent import KnowledgeBase, QueryProcessor, UsageAnalytics, Query

kb = KnowledgeBase()
# ... load documents or add sources ...
analytics = UsageAnalytics()
processor = QueryProcessor(kb, analytics=analytics)

results = processor.process_query(Query(text="deployment process", user="U1"))
for doc in results:
    print(doc.content)
```

## Routing Unanswered Questions

```python
from slack_kb_agent import RoutingEngine, TeamMember

members = [TeamMember(id="U1", name="Alice", expertise=["backend"])]
router = RoutingEngine(members)
processor = QueryProcessor(kb, routing=router)

_, experts = processor.search_and_route("backend issue")
if experts:
    print("Escalate to", experts[0].name)
```

## Accessing Usage Analytics

```python
# After processing queries
print("Top queries:", analytics.top_queries())
print("Top users:", analytics.top_users())
print("Top channels:", analytics.top_channels())

# Persist analytics to a file
analytics.save("analytics.json")

# Load analytics later
loaded = UsageAnalytics.load("analytics.json")
```

## Persisting the Knowledge Base

```python
kb.save("kb.json")
loaded_kb = KnowledgeBase.load("kb.json")
```

## Command Line Usage

```bash
$ slack-kb-agent "search term" --kb kb.json --analytics analytics.json
```

For more details on configuration and deployment, see `README.md`.
