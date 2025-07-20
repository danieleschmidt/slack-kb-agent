# Changelog

## v1.6.4 - 2025-07-20

- feat(cache): implement comprehensive Redis-based caching layer for performance optimization
- feat(cache): add embedding caching for vector search with configurable TTL (7 days default)
- feat(cache): add query expansion caching for synonyms, technical terms, and LLM-based expansion
- feat(cache): add search results caching with automatic invalidation on document updates
- feat(cache): integrate cache metrics into monitoring system (hit rates, key counts, availability)
- feat(cache): add cache invalidation on knowledge base document additions
- feat(cache): graceful fallback when Redis unavailable with comprehensive error handling
- feat(monitoring): add cache metrics to health checks and Prometheus metrics
- tests: comprehensive test coverage for cache functionality and integration
- docs: update backlog with completed caching layer implementation

## v1.6.3 - 2025-07-20

- feat(tests): implement comprehensive HTTP server integration tests for authentication
- feat(tests): add tests for basic auth, API key auth, rate limiting, and mixed auth methods
- feat(auth): fix case-insensitive API key header handling for HTTP server compatibility
- feat(tests): add real HTTP server testing with standard library (no external dependencies)
- feat(tests): comprehensive test coverage for all authentication scenarios
- tests: verify audit logging and error response validation in HTTP integration tests
- docs: update backlog with completed HTTP server integration tests

## v1.6.2 - 2025-07-20

- feat(exceptions): create comprehensive exception hierarchy for better error handling
- feat(monitoring): replace broad exception handlers with specific error types
- feat(monitoring): add detailed error logging with context information
- feat(monitoring): implement error metrics for monitoring failure patterns
- feat(monitoring): add "unknown" health status for unreachable system checks
- feat(monitoring): enhance health check priority system
- tests: comprehensive test coverage for error handling scenarios
- docs: update backlog with completed error handling improvements

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
