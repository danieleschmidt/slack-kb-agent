# Changelog

## v1.7.5 - 2025-07-24

- **SECURITY FIX**: Fix critical SQL injection vulnerability in search_documents method
- feat(security): replace f-string SQL interpolation with SQLAlchemy's func.concat for safe parameter binding
- feat(security): add comprehensive input validation and sanitization for search queries
- feat(security): implement query length limits to prevent DoS attacks
- feat(security): enhance error handling with proper logging for search operations
- feat(testing): add comprehensive SQL injection security test suite
- refactor(database): improve search functionality while maintaining performance
- security(database): prevent arbitrary SQL execution through search parameters
- **TESTING**: Add comprehensive test coverage for security_utils module (40 test cases)
- feat(testing): test credential masking for database URLs, connection strings, and dictionaries
- feat(testing): verify SQL injection protection and input validation
- feat(testing): ensure sensitive data protection across all utility functions
- fix(security): improve complex password parsing with @ symbols in passwords
- fix(security): enhance regex patterns for more accurate credential detection
- refactor(security): strengthen connection string detection and masking logic

## v1.7.4 - 2025-07-22

- feat(error-handling): improve cache exception handling with better categorization
- feat(error-handling): add specific handling for MemoryError, ConnectionError, TimeoutError in cache operations
- feat(error-handling): enhance error messages with error type and specific context for better debugging
- feat(testing): add comprehensive cache exception handling test suite
- feat(testing): create empty exception handler analysis tool for quality verification
- feat(observability): improve error categorization for production cache monitoring
- refactor(cache): maintain defensive programming patterns while enhancing error specificity
- feat(resilience): ensure cache failures never crash main application with graceful fallback

## v1.7.3 - 2025-07-22

- feat(config): extract hardcoded constants to centralized configuration module
- feat(config): create comprehensive constants.py with organized configuration classes
- feat(config): standardize network settings (ports, timeouts, connection limits)
- feat(config): centralize rate limiting values with environment variable support
- feat(config): consolidate validation limits and monitoring thresholds
- feat(config): add environment-aware configuration with validation and range checking
- feat(maintainability): eliminate magic numbers throughout codebase
- feat(error-handling): replace broad exception handlers with specific types in core modules
- feat(error-handling): enhanced Slack bot error handling with specific data validation errors  
- feat(error-handling): improved LLM provider error handling with connection/parsing error types
- feat(error-handling): added proper error context and logging for better debugging capabilities
- feat(error-handling): implemented defensive catch-all handlers to prevent service crashes
- feat(testing): added comprehensive exception handler analysis tool (test_specific_exception_handling.py)
- feat(observability): improved error categorization for better production monitoring
- refactor(backlog): updated current iteration backlog with detailed progress tracking
- docs: enhanced error handling documentation with specific error types and contexts

## v1.7.2 - 2025-07-21

- fix(security): implement secure bcrypt password hashing to replace plaintext password storage
- fix(security): add automatic migration of existing plaintext passwords to secure hashes
- feat(auth): create PasswordHasher class with configurable cost parameters and timing-safe verification
- feat(auth): add BasicAuthenticator class with integrated password hashing and validation
- feat(auth): implement salt randomness and timing attack resistance
- feat(security): add comprehensive password validation with special character and long password support
- tests: add extensive test coverage for password hashing security properties and edge cases
- tests: verify timing attack resistance and salt randomness properties
- deps: add bcrypt>=4.0.0 dependency for cryptographically secure password hashing
- docs: update authentication documentation with security best practices

## v1.7.1 - 2025-07-21

- fix(security): remove hard-coded database credentials from all configuration files
- fix(security): require DATABASE_URL environment variable with helpful error messages
- feat(search): implement high-performance inverted index search engine with TF-IDF scoring
- feat(search): replace O(n) linear search with O(log n) indexed search for massive performance improvement
- feat(search): add search result caching with LRU eviction for frequently accessed queries
- feat(search): implement phrase matching bonuses and relevance scoring
- feat(search): add comprehensive search statistics and performance monitoring
- feat(search): maintain backward compatibility with existing search API
- tests: add comprehensive test suite for indexed search functionality and performance
- tests: add security tests to prevent regression of hard-coded credentials
- docs: update README with enhanced search capabilities and security best practices

## v1.7.0 - 2025-07-21

- feat(database): implement comprehensive PostgreSQL database persistence with SQLAlchemy
- feat(database): add production-ready connection pooling and environment-based configuration
- feat(database): create Alembic-based schema migrations with version control
- feat(database): implement enhanced PersistentKnowledgeBase with hybrid storage (database + in-memory)
- feat(database): add lazy loading from database with automatic fallback to JSON persistence
- feat(backup): implement comprehensive backup/restore system with compression and validation
- feat(backup): add gzip compression support and JSON backup compatibility
- feat(backup): create backup validation with detailed error reporting and metadata tracking
- feat(cli): add database management CLI tool (`slack-kb-db`) for all database operations
- feat(cli): support database initialization, backup, restore, migration, and status checking
- feat(migrations): create initial database schema migration with proper indexing
- feat(persistence): maintain backward compatibility with existing JSON file persistence
- feat(monitoring): integrate database statistics into monitoring and health checks
- tests: comprehensive test coverage for database, backup, and persistent knowledge base functionality
- docs: update README with database setup, management commands, and architecture diagrams
- deps: add SQLAlchemy, psycopg2-binary, and Alembic dependencies for PostgreSQL support

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
