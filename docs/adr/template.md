# ADR-XXX: [Decision Title]

**Status**: [Proposed | Accepted | Deprecated | Superseded]  
**Date**: [YYYY-MM-DD]  
**Deciders**: [List of decision makers]  
**Technical Story**: [Link to issue/ticket]

## Context and Problem Statement

[Describe the context and problem statement, e.g., in free form using two to three sentences. You may want to articulate the problem in form of a question.]

## Decision Drivers

* [driver 1, e.g., a force, facing concern, …]
* [driver 2, e.g., a force, facing concern, …]
* [… numbers of drivers can vary]

## Considered Options

* [option 1]
* [option 2]
* [option 3]
* [… numbers of options can vary]

## Decision Outcome

Chosen option: "[option 1]", because [justification. e.g., only option, which meets k.o. criterion decision driver | which resolves force force | … | comes out best (see below)].

### Positive Consequences

* [e.g., improvement of quality attribute satisfaction, follow-up decisions required, …]
* [… numbers can vary]

### Negative Consequences

* [e.g., compromising quality attribute, follow-up decisions required, …]
* [… numbers can vary]

## Pros and Cons of the Options

### [option 1]

[example | description | pointer to more information | …]

* Good, because [argument a]
* Good, because [argument b]
* Bad, because [argument c]
* [… numbers of pros and cons can vary]

### [option 2]

[example | description | pointer to more information | …]

* Good, because [argument a]
* Good, because [argument b]
* Bad, because [argument c]
* [… numbers of pros and cons can vary]

### [option 3]

[example | description | pointer to more information | …]

* Good, because [argument a]
* Good, because [argument b]
* Bad, because [argument c]
* [… numbers of pros and cons can vary]

## Links

* [Link type] [Link to ADR] <!-- example: Refines [ADR-0005](0005-example.md) -->
* [ADR-0000](0000-use-markdown-architectural-decision-records.md) <!-- example: -->
* [… numbers of links can vary]

## Implementation Notes

[Optional section for implementation details, migration steps, or follow-up actions]

### Migration Plan
* [Step 1]
* [Step 2]
* [… additional steps as needed]

### Success Criteria
* [Criterion 1]
* [Criterion 2]
* [… additional criteria as needed]

### Monitoring and Review
* [How will we know this decision is working?]
* [When should we review this decision?]
* [What metrics will we track?]

---

## Template Usage Notes

1. **Copy this template** for each new ADR
2. **Number sequentially**: ADR-001, ADR-002, etc.
3. **Use descriptive titles**: Clear, concise, action-oriented
4. **Update status**: Start with "Proposed", move to "Accepted" when implemented
5. **Link related ADRs**: Reference decisions that build on or supersede this one
6. **Keep it concise**: Aim for 1-2 pages maximum
7. **Include implementation details**: Help future developers understand the decision

### Example Titles
* ADR-001: Database Selection for Knowledge Storage
* ADR-002: Vector Search Implementation Strategy
* ADR-003: Authentication and Authorization Approach
* ADR-004: Caching Strategy for Performance Optimization
* ADR-005: Error Handling and Circuit Breaker Patterns

### Status Lifecycle
* **Proposed**: Decision is being considered
* **Accepted**: Decision has been made and is being implemented
* **Deprecated**: Decision is no longer recommended but may still be in use
* **Superseded**: Decision has been replaced by a newer ADR (reference the new one)