# Changelog

## v1.6.1 - 2025-07-20

- feat(memory): implement comprehensive memory management for production stability
- feat(memory): add configurable document limits to KnowledgeBase with FIFO eviction
- feat(memory): implement LRU eviction for user conversation contexts
- feat(memory): add TTL-based cleanup for rate limiter request tracking
- feat(monitoring): add memory usage metrics collection and reporting
- feat(monitoring): add memory stats APIs for all major components
- tests: comprehensive test coverage for memory management features
- docs: update backlog with completed memory management implementation

## v1.3.0 - 2025-07-19

- feat(search): implement vector-based semantic search with FAISS
- feat(search): add hybrid search combining semantic and keyword approaches
- feat(search): configurable similarity thresholds and model selection
- feat(search): automatic fallback to keyword search when dependencies unavailable
- docs: comprehensive vector search API documentation and examples
- tests: extensive test coverage with mocked dependencies
- chore: add WSJF-prioritized development backlog

## v1.2.0 - 2025-06-27

- feat(cli): add command line interface for querying the knowledge base
- docs: document CLI usage in README and API guide
- tests: cover CLI functionality

## v1.1.0 - 2025-06-27

- feat(kb): allow saving and loading knowledge base documents
- docs: note knowledge base persistence in README and API guide
- tests: add coverage for knowledge base persistence

## v1.0.1 - 2025-06-27

- feat(analytics): allow saving and loading analytics from JSON
- docs: mention analytics persistence in README and API guide
- tests: cover analytics persistence

## v1.0.0 - 2025-06-27

- first stable release with analytics and smart routing
- docs: API usage guide and README updates


## v0.0.7 - 2025-06-27

- docs: add API usage guide and reference from README

## v0.0.6 - 2025-06-27

- feat(analytics): track query counts per channel via ``UsageAnalytics.top_channels``
- docs: mention channel tracking in README

## v0.0.5 - 2025-06-27

- feat(analytics): track query counts per user via ``UsageAnalytics.top_users``
- docs: mention analytics now reports active users

## v0.0.4 - 2025-06-27

- feat(analytics): integrate ``UsageAnalytics`` with ``QueryProcessor``

## v0.0.3 - 2025-06-26

- feat(analytics): add simple UsageAnalytics module
- docs: mention analytics in README

## v0.0.2 - 2025-06-25

- Applying previous commit. (0d0571f)
- Merge pull request #1 from danieleschmidt/codex/generate/update-strategic-development-plan (14ddbaa)
- docs(review): add code review report (42f278b)
- Update README.md (10cdb06)
- Initial commit (9096741)
