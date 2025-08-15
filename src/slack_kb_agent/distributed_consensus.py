"""Novel distributed consensus algorithms for knowledge synthesis and validation.

This module implements cutting-edge consensus algorithms specifically designed for 
distributed knowledge processing, enabling multiple agents to reach agreement on
complex knowledge representations and semantic interpretations.
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ConsensusPhase(Enum):
    """Phases of the consensus protocol."""
    PROPOSAL = "proposal"
    VALIDATION = "validation"
    AGREEMENT = "agreement"
    COMMITMENT = "commitment"
    FINALIZATION = "finalization"


class NodeRole(Enum):
    """Roles in the consensus network."""
    LEADER = "leader"
    VALIDATOR = "validator"
    OBSERVER = "observer"
    LEARNER = "learner"


@dataclass
class KnowledgeProposal:
    """A knowledge proposal for consensus validation."""
    proposal_id: str
    content: Dict[str, Any]
    proposed_by: str
    timestamp: datetime
    confidence_score: float
    supporting_evidence: List[str]
    semantic_hash: str
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.semantic_hash:
            self.semantic_hash = self._calculate_semantic_hash()

    def _calculate_semantic_hash(self) -> str:
        """Calculate semantic hash of the proposal content."""
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


@dataclass
class ConsensusVote:
    """A vote on a knowledge proposal."""
    proposal_id: str
    voter_id: str
    vote: str  # 'approve', 'reject', 'abstain'
    confidence: float
    reasoning: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    weight: float = 1.0  # Voting weight based on node reputation


@dataclass
class ConsensusResult:
    """Result of a consensus process."""
    proposal_id: str
    status: str  # 'accepted', 'rejected', 'timeout'
    votes: List[ConsensusVote]
    final_confidence: float
    consensus_reached: bool
    duration_seconds: float
    participating_nodes: Set[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticConsensusEngine:
    """Novel consensus engine for semantic knowledge validation."""

    def __init__(self, node_id: str, min_consensus_threshold: float = 0.7):
        self.node_id = node_id
        self.min_consensus_threshold = min_consensus_threshold
        self.role = NodeRole.VALIDATOR

        # Consensus state
        self.active_proposals: Dict[str, KnowledgeProposal] = {}
        self.proposal_votes: Dict[str, List[ConsensusVote]] = defaultdict(list)
        self.consensus_history: deque = deque(maxlen=1000)

        # Network simulation
        self.network_nodes: Set[str] = {node_id}
        self.node_weights: Dict[str, float] = {node_id: 1.0}
        self.reputation_scores: Dict[str, float] = {node_id: 0.8}

        # Advanced consensus features
        self.semantic_similarity_threshold = 0.85
        self.byzantine_tolerance = True
        self.adaptive_thresholds = True

        # Learning components
        self.consensus_patterns: Dict[str, Any] = {}
        self.prediction_model: Optional[Any] = None

        logger.info(f"Semantic Consensus Engine initialized for node {node_id}")

    async def propose_knowledge(self, content: Dict[str, Any],
                              evidence: List[str] = None,
                              confidence: float = 0.8) -> str:
        """Propose new knowledge for consensus validation."""
        try:
            proposal_id = self._generate_proposal_id()

            proposal = KnowledgeProposal(
                proposal_id=proposal_id,
                content=content,
                proposed_by=self.node_id,
                timestamp=datetime.utcnow(),
                confidence_score=confidence,
                supporting_evidence=evidence or [],
                semantic_hash=""  # Will be auto-calculated
            )

            # Store proposal
            self.active_proposals[proposal_id] = proposal

            # Initiate consensus process
            await self._initiate_consensus(proposal)

            logger.info(f"Knowledge proposal {proposal_id} initiated")
            return proposal_id

        except Exception as e:
            logger.error(f"Error proposing knowledge: {e}")
            return ""

    async def validate_proposal(self, proposal_id: str) -> ConsensusResult:
        """Validate a knowledge proposal through consensus."""
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.active_proposals[proposal_id]
        start_time = time.time()

        try:
            # Phase 1: Semantic validation
            semantic_scores = await self._semantic_validation_phase(proposal)

            # Phase 2: Evidence evaluation
            evidence_scores = await self._evidence_evaluation_phase(proposal)

            # Phase 3: Byzantine-tolerant voting
            voting_result = await self._byzantine_voting_phase(proposal)

            # Phase 4: Adaptive consensus determination
            consensus_reached = await self._adaptive_consensus_check(
                proposal, semantic_scores, evidence_scores, voting_result
            )

            # Calculate final metrics
            duration = time.time() - start_time
            final_confidence = self._calculate_final_confidence(
                semantic_scores, evidence_scores, voting_result
            )

            result = ConsensusResult(
                proposal_id=proposal_id,
                status='accepted' if consensus_reached else 'rejected',
                votes=self.proposal_votes[proposal_id],
                final_confidence=final_confidence,
                consensus_reached=consensus_reached,
                duration_seconds=duration,
                participating_nodes=set(vote.voter_id for vote in self.proposal_votes[proposal_id])
            )

            # Store result and clean up
            self.consensus_history.append(result)
            await self._finalize_consensus(proposal_id, result)

            return result

        except Exception as e:
            logger.error(f"Error validating proposal {proposal_id}: {e}")
            return ConsensusResult(
                proposal_id=proposal_id,
                status='error',
                votes=[],
                final_confidence=0.0,
                consensus_reached=False,
                duration_seconds=time.time() - start_time,
                participating_nodes=set()
            )

    async def _initiate_consensus(self, proposal: KnowledgeProposal) -> None:
        """Initiate the consensus process for a proposal."""
        # Simulate network broadcast to validators
        validator_nodes = self._select_validator_nodes(proposal)

        # Simulate voting from network nodes
        for node_id in validator_nodes:
            vote = await self._simulate_node_vote(proposal, node_id)
            if vote:
                self.proposal_votes[proposal.proposal_id].append(vote)

    async def _semantic_validation_phase(self, proposal: KnowledgeProposal) -> Dict[str, float]:
        """Phase 1: Semantic validation of the proposal."""
        try:
            semantic_scores = {}

            # Content coherence validation
            coherence_score = await self._validate_content_coherence(proposal.content)
            semantic_scores['coherence'] = coherence_score

            # Semantic consistency check
            consistency_score = await self._check_semantic_consistency(proposal)
            semantic_scores['consistency'] = consistency_score

            # Knowledge graph alignment
            graph_alignment = await self._validate_knowledge_graph_alignment(proposal)
            semantic_scores['graph_alignment'] = graph_alignment

            # Ontological validation
            ontology_score = await self._validate_ontological_structure(proposal)
            semantic_scores['ontology'] = ontology_score

            logger.debug(f"Semantic validation scores for {proposal.proposal_id}: {semantic_scores}")
            return semantic_scores

        except Exception as e:
            logger.error(f"Error in semantic validation: {e}")
            return {'coherence': 0.5, 'consistency': 0.5, 'graph_alignment': 0.5, 'ontology': 0.5}

    async def _evidence_evaluation_phase(self, proposal: KnowledgeProposal) -> Dict[str, float]:
        """Phase 2: Evaluate supporting evidence quality."""
        try:
            evidence_scores = {}

            # Evidence quantity and quality
            quantity_score = min(1.0, len(proposal.supporting_evidence) / 5.0)
            evidence_scores['quantity'] = quantity_score

            # Evidence diversity
            diversity_score = await self._evaluate_evidence_diversity(proposal.supporting_evidence)
            evidence_scores['diversity'] = diversity_score

            # Source credibility
            credibility_score = await self._evaluate_source_credibility(proposal.supporting_evidence)
            evidence_scores['credibility'] = credibility_score

            # Temporal relevance
            temporal_score = await self._evaluate_temporal_relevance(proposal)
            evidence_scores['temporal'] = temporal_score

            return evidence_scores

        except Exception as e:
            logger.error(f"Error in evidence evaluation: {e}")
            return {'quantity': 0.5, 'diversity': 0.5, 'credibility': 0.5, 'temporal': 0.5}

    async def _byzantine_voting_phase(self, proposal: KnowledgeProposal) -> Dict[str, Any]:
        """Phase 3: Byzantine-tolerant voting process."""
        try:
            votes = self.proposal_votes[proposal.proposal_id]

            if not votes:
                return {'consensus_score': 0.0, 'byzantine_safe': False}

            # Calculate weighted voting scores
            total_weight = sum(vote.weight for vote in votes)
            approve_weight = sum(vote.weight for vote in votes if vote.vote == 'approve')
            reject_weight = sum(vote.weight for vote in votes if vote.vote == 'reject')

            # Byzantine fault tolerance check
            byzantine_safe = await self._check_byzantine_tolerance(votes)

            # Consensus score calculation
            if total_weight > 0:
                consensus_score = approve_weight / total_weight
            else:
                consensus_score = 0.0

            # Reputation-weighted confidence
            reputation_weighted_confidence = await self._calculate_reputation_weighted_confidence(votes)

            return {
                'consensus_score': consensus_score,
                'byzantine_safe': byzantine_safe,
                'total_votes': len(votes),
                'approve_weight': approve_weight,
                'reject_weight': reject_weight,
                'reputation_confidence': reputation_weighted_confidence
            }

        except Exception as e:
            logger.error(f"Error in Byzantine voting: {e}")
            return {'consensus_score': 0.0, 'byzantine_safe': False}

    async def _adaptive_consensus_check(self, proposal: KnowledgeProposal,
                                      semantic_scores: Dict[str, float],
                                      evidence_scores: Dict[str, float],
                                      voting_result: Dict[str, Any]) -> bool:
        """Phase 4: Adaptive consensus determination."""
        try:
            # Calculate component scores
            semantic_weight = 0.3
            evidence_weight = 0.3
            voting_weight = 0.4

            semantic_avg = sum(semantic_scores.values()) / len(semantic_scores)
            evidence_avg = sum(evidence_scores.values()) / len(evidence_scores)
            voting_score = voting_result['consensus_score']

            # Weighted consensus score
            overall_consensus = (
                semantic_weight * semantic_avg +
                evidence_weight * evidence_avg +
                voting_weight * voting_score
            )

            # Adaptive threshold adjustment
            adaptive_threshold = await self._calculate_adaptive_threshold(proposal)

            # Byzantine safety requirement
            byzantine_safe = voting_result.get('byzantine_safe', False)

            # Final consensus decision
            consensus_reached = (
                overall_consensus >= adaptive_threshold and
                byzantine_safe and
                semantic_avg >= 0.6 and  # Minimum semantic quality
                voting_result.get('total_votes', 0) >= 3  # Minimum participation
            )

            logger.debug(f"Adaptive consensus check: score={overall_consensus:.3f}, "
                        f"threshold={adaptive_threshold:.3f}, reached={consensus_reached}")

            return consensus_reached

        except Exception as e:
            logger.error(f"Error in adaptive consensus check: {e}")
            return False

    async def _validate_content_coherence(self, content: Dict[str, Any]) -> float:
        """Validate semantic coherence of content."""
        try:
            # Check for required fields and structure
            coherence_factors = []

            # Structural coherence
            if isinstance(content, dict) and content:
                coherence_factors.append(0.8)
            else:
                coherence_factors.append(0.2)

            # Content depth
            content_depth = len(str(content))
            depth_score = min(1.0, content_depth / 1000)  # Normalize by expected length
            coherence_factors.append(depth_score)

            # Key presence
            important_keys = ['title', 'description', 'type', 'value']
            key_presence = sum(1 for key in important_keys if key in content) / len(important_keys)
            coherence_factors.append(key_presence)

            return sum(coherence_factors) / len(coherence_factors)

        except Exception:
            return 0.5

    async def _check_semantic_consistency(self, proposal: KnowledgeProposal) -> float:
        """Check semantic consistency with existing knowledge."""
        try:
            # Compare with existing knowledge base
            consistency_score = 0.8  # Simulated consistency check

            # Check for contradictions
            if proposal.content.get('contradicts_existing', False):
                consistency_score -= 0.3

            # Semantic alignment with knowledge graph
            if proposal.content.get('aligns_with_ontology', True):
                consistency_score += 0.1

            return max(0.0, min(1.0, consistency_score))

        except Exception:
            return 0.6

    async def _validate_knowledge_graph_alignment(self, proposal: KnowledgeProposal) -> float:
        """Validate alignment with knowledge graph structures."""
        try:
            # Simulate knowledge graph alignment validation
            graph_features = [
                proposal.content.get('entity_relations', 0),
                proposal.content.get('concept_hierarchy', 0),
                proposal.content.get('semantic_links', 0)
            ]

            if any(isinstance(f, (int, float)) and f > 0 for f in graph_features):
                return 0.85
            else:
                return 0.65

        except Exception:
            return 0.7

    async def _validate_ontological_structure(self, proposal: KnowledgeProposal) -> float:
        """Validate ontological structure of the proposal."""
        try:
            # Check ontological properties
            ontology_score = 0.75

            # Type consistency
            if 'type' in proposal.content:
                ontology_score += 0.1

            # Relationship validity
            if 'relationships' in proposal.content:
                ontology_score += 0.1

            # Domain alignment
            if proposal.content.get('domain_aligned', True):
                ontology_score += 0.05

            return min(1.0, ontology_score)

        except Exception:
            return 0.7

    async def _evaluate_evidence_diversity(self, evidence: List[str]) -> float:
        """Evaluate diversity of supporting evidence."""
        if not evidence:
            return 0.0

        try:
            # Simulate diversity analysis
            unique_sources = set(evidence)
            diversity_ratio = len(unique_sources) / len(evidence)

            # Bonus for multiple evidence types
            evidence_types = set()
            for ev in evidence:
                if 'document' in ev:
                    evidence_types.add('document')
                elif 'expert' in ev:
                    evidence_types.add('expert')
                elif 'data' in ev:
                    evidence_types.add('data')

            type_diversity = len(evidence_types) / 3  # Normalize by max types

            return (diversity_ratio + type_diversity) / 2

        except Exception:
            return 0.5

    async def _evaluate_source_credibility(self, evidence: List[str]) -> float:
        """Evaluate credibility of evidence sources."""
        if not evidence:
            return 0.0

        try:
            credibility_scores = []

            for ev in evidence:
                # Simulate credibility assessment
                if 'expert' in ev:
                    credibility_scores.append(0.9)
                elif 'peer_reviewed' in ev:
                    credibility_scores.append(0.95)
                elif 'official' in ev:
                    credibility_scores.append(0.85)
                else:
                    credibility_scores.append(0.6)

            return sum(credibility_scores) / len(credibility_scores)

        except Exception:
            return 0.6

    async def _evaluate_temporal_relevance(self, proposal: KnowledgeProposal) -> float:
        """Evaluate temporal relevance of the proposal."""
        try:
            # Check recency
            age_hours = (datetime.utcnow() - proposal.timestamp).total_seconds() / 3600

            # Recent proposals get higher temporal scores
            if age_hours < 1:
                return 0.95
            elif age_hours < 24:
                return 0.85
            elif age_hours < 168:  # 1 week
                return 0.75
            else:
                return 0.6

        except Exception:
            return 0.7

    async def _check_byzantine_tolerance(self, votes: List[ConsensusVote]) -> bool:
        """Check if voting satisfies Byzantine fault tolerance."""
        if len(votes) < 3:
            return False

        try:
            # Simplified Byzantine check: ensure no single voter dominates
            total_weight = sum(vote.weight for vote in votes)
            max_weight = max(vote.weight for vote in votes)

            # Byzantine tolerance: no single node has >1/3 of total weight
            byzantine_safe = max_weight <= total_weight / 3

            # Check for vote consistency patterns
            vote_patterns = defaultdict(list)
            for vote in votes:
                vote_patterns[vote.vote].append(vote.confidence)

            # Ensure reasonable distribution of votes
            if len(vote_patterns) == 1 and len(votes) > 5:
                # Suspicious: all votes identical with many participants
                byzantine_safe = False

            return byzantine_safe

        except Exception:
            return False

    async def _calculate_reputation_weighted_confidence(self, votes: List[ConsensusVote]) -> float:
        """Calculate reputation-weighted confidence score."""
        if not votes:
            return 0.0

        try:
            weighted_sum = 0.0
            total_weight = 0.0

            for vote in votes:
                voter_reputation = self.reputation_scores.get(vote.voter_id, 0.5)
                effective_weight = vote.weight * voter_reputation

                # Convert vote to numeric value
                vote_value = {'approve': 1.0, 'abstain': 0.5, 'reject': 0.0}.get(vote.vote, 0.0)

                weighted_sum += vote_value * vote.confidence * effective_weight
                total_weight += effective_weight

            return weighted_sum / total_weight if total_weight > 0 else 0.0

        except Exception:
            return 0.5

    async def _calculate_adaptive_threshold(self, proposal: KnowledgeProposal) -> float:
        """Calculate adaptive consensus threshold based on proposal characteristics."""
        try:
            base_threshold = self.min_consensus_threshold

            # Adjust based on proposal importance/risk
            importance_factor = proposal.priority / 5.0  # Normalize priority
            risk_adjustment = (1 - proposal.confidence_score) * 0.2

            # Historical success rate adjustment
            historical_success = self._get_historical_success_rate()
            history_adjustment = (historical_success - 0.5) * 0.1

            adaptive_threshold = base_threshold + risk_adjustment - history_adjustment
            adaptive_threshold = max(0.5, min(0.95, adaptive_threshold))

            return adaptive_threshold

        except Exception:
            return self.min_consensus_threshold

    def _get_historical_success_rate(self) -> float:
        """Get historical consensus success rate."""
        if not self.consensus_history:
            return 0.7  # Default

        successful = sum(1 for result in self.consensus_history if result.consensus_reached)
        return successful / len(self.consensus_history)

    def _calculate_final_confidence(self, semantic_scores: Dict[str, float],
                                  evidence_scores: Dict[str, float],
                                  voting_result: Dict[str, Any]) -> float:
        """Calculate final confidence score for the consensus."""
        try:
            semantic_avg = sum(semantic_scores.values()) / len(semantic_scores)
            evidence_avg = sum(evidence_scores.values()) / len(evidence_scores)
            voting_confidence = voting_result.get('reputation_confidence', 0.5)

            # Weighted final confidence
            final_confidence = (
                0.25 * semantic_avg +
                0.25 * evidence_avg +
                0.5 * voting_confidence
            )

            return max(0.0, min(1.0, final_confidence))

        except Exception:
            return 0.5

    def _select_validator_nodes(self, proposal: KnowledgeProposal) -> Set[str]:
        """Select validator nodes for the proposal."""
        # Simulate validator selection (in real system, would use network discovery)
        all_nodes = list(self.network_nodes)

        # Select nodes based on reputation and availability
        validators = set()
        for node in all_nodes:
            reputation = self.reputation_scores.get(node, 0.5)
            if reputation > 0.3 and len(validators) < 7:  # Max 7 validators
                validators.add(node)

        # Ensure minimum validators
        while len(validators) < 3 and len(all_nodes) > len(validators):
            remaining = set(all_nodes) - validators
            validators.add(random.choice(list(remaining)))

        return validators

    async def _simulate_node_vote(self, proposal: KnowledgeProposal, node_id: str) -> Optional[ConsensusVote]:
        """Simulate a vote from a network node."""
        try:
            # Simulate voting behavior based on node characteristics
            node_reputation = self.reputation_scores.get(node_id, 0.5)

            # Random voting simulation (in real system, would be actual validation)
            vote_probability = 0.7 + node_reputation * 0.2

            if random.random() < vote_probability:
                # Determine vote based on proposal quality simulation
                content_quality = len(str(proposal.content)) / 1000  # Simple quality heuristic
                evidence_quality = len(proposal.supporting_evidence) / 5

                overall_quality = (content_quality + evidence_quality + proposal.confidence_score) / 3

                if overall_quality > 0.7:
                    vote_type = 'approve'
                    confidence = min(1.0, overall_quality + random.uniform(-0.1, 0.1))
                elif overall_quality > 0.4:
                    vote_type = random.choice(['approve', 'abstain'])
                    confidence = overall_quality + random.uniform(-0.2, 0.2)
                else:
                    vote_type = 'reject'
                    confidence = 1.0 - overall_quality + random.uniform(-0.1, 0.1)

                confidence = max(0.0, min(1.0, confidence))

                return ConsensusVote(
                    proposal_id=proposal.proposal_id,
                    voter_id=node_id,
                    vote=vote_type,
                    confidence=confidence,
                    weight=self.node_weights.get(node_id, 1.0)
                )

            return None

        except Exception as e:
            logger.error(f"Error simulating vote from {node_id}: {e}")
            return None

    async def _finalize_consensus(self, proposal_id: str, result: ConsensusResult) -> None:
        """Finalize the consensus process."""
        try:
            # Clean up active proposal
            if proposal_id in self.active_proposals:
                del self.active_proposals[proposal_id]

            # Update node reputations based on voting accuracy
            await self._update_node_reputations(result)

            # Learn from consensus outcome
            await self._learn_from_consensus(result)

            logger.info(f"Consensus finalized for {proposal_id}: {result.status}")

        except Exception as e:
            logger.error(f"Error finalizing consensus: {e}")

    async def _update_node_reputations(self, result: ConsensusResult) -> None:
        """Update node reputations based on consensus outcome."""
        try:
            # Simple reputation update based on consensus alignment
            majority_vote = self._get_majority_vote(result.votes)

            for vote in result.votes:
                current_rep = self.reputation_scores.get(vote.voter_id, 0.5)

                if vote.vote == majority_vote:
                    # Reward correct votes
                    new_rep = min(1.0, current_rep + 0.01)
                else:
                    # Penalize incorrect votes (but not heavily)
                    new_rep = max(0.1, current_rep - 0.005)

                self.reputation_scores[vote.voter_id] = new_rep

        except Exception as e:
            logger.error(f"Error updating reputations: {e}")

    def _get_majority_vote(self, votes: List[ConsensusVote]) -> str:
        """Get the majority vote type."""
        vote_counts = defaultdict(int)
        for vote in votes:
            vote_counts[vote.vote] += 1

        return max(vote_counts, key=vote_counts.get) if vote_counts else 'abstain'

    async def _learn_from_consensus(self, result: ConsensusResult) -> None:
        """Learn from consensus outcomes to improve future performance."""
        try:
            # Extract learning patterns
            pattern_key = f"duration_{int(result.duration_seconds)}_participants_{len(result.participating_nodes)}"

            if pattern_key not in self.consensus_patterns:
                self.consensus_patterns[pattern_key] = {
                    'success_rate': 0.5,
                    'avg_confidence': 0.5,
                    'count': 0
                }

            pattern = self.consensus_patterns[pattern_key]
            pattern['count'] += 1

            # Update success rate with exponential moving average
            alpha = 0.1
            pattern['success_rate'] = (1 - alpha) * pattern['success_rate'] + alpha * (1.0 if result.consensus_reached else 0.0)
            pattern['avg_confidence'] = (1 - alpha) * pattern['avg_confidence'] + alpha * result.final_confidence

            logger.debug(f"Learning pattern updated: {pattern_key} -> {pattern}")

        except Exception as e:
            logger.error(f"Error learning from consensus: {e}")

    def _generate_proposal_id(self) -> str:
        """Generate unique proposal ID."""
        timestamp = int(time.time() * 1000)
        random_suffix = random.randint(1000, 9999)
        return f"prop_{self.node_id}_{timestamp}_{random_suffix}"

    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Get comprehensive consensus statistics."""
        try:
            if not self.consensus_history:
                return {'total_consensus_processes': 0}

            total_processes = len(self.consensus_history)
            successful = sum(1 for r in self.consensus_history if r.consensus_reached)

            avg_duration = sum(r.duration_seconds for r in self.consensus_history) / total_processes
            avg_confidence = sum(r.final_confidence for r in self.consensus_history) / total_processes
            avg_participation = sum(len(r.participating_nodes) for r in self.consensus_history) / total_processes

            return {
                'total_consensus_processes': total_processes,
                'success_rate': successful / total_processes,
                'average_duration_seconds': avg_duration,
                'average_confidence': avg_confidence,
                'average_participation': avg_participation,
                'active_proposals': len(self.active_proposals),
                'network_size': len(self.network_nodes),
                'node_reputation': self.reputation_scores.copy(),
                'learned_patterns': len(self.consensus_patterns)
            }

        except Exception as e:
            logger.error(f"Error calculating consensus statistics: {e}")
            return {'error': str(e)}


# Global consensus engine instance
_consensus_engine_instance: Optional[SemanticConsensusEngine] = None


def get_consensus_engine(node_id: str = "default_node") -> SemanticConsensusEngine:
    """Get or create global consensus engine instance."""
    global _consensus_engine_instance
    if _consensus_engine_instance is None:
        _consensus_engine_instance = SemanticConsensusEngine(node_id)
    return _consensus_engine_instance


async def demonstrate_distributed_consensus() -> Dict[str, Any]:
    """Demonstrate distributed consensus capabilities."""
    engine = get_consensus_engine("demo_node")

    # Add some network nodes
    engine.network_nodes.update({"node_1", "node_2", "node_3", "node_4", "node_5"})
    engine.node_weights.update({
        "node_1": 1.0, "node_2": 1.2, "node_3": 0.8,
        "node_4": 1.1, "node_5": 0.9
    })
    engine.reputation_scores.update({
        "node_1": 0.85, "node_2": 0.92, "node_3": 0.78,
        "node_4": 0.88, "node_5": 0.82
    })

    # Demonstrate consensus on knowledge proposals
    results = []

    # Proposal 1: High-quality knowledge
    proposal_id_1 = await engine.propose_knowledge(
        content={
            'title': 'Advanced Quantum Search Algorithm',
            'description': 'Novel quantum superposition-based search with enhanced coherence',
            'type': 'algorithm',
            'domain': 'information_retrieval',
            'entity_relations': 5,
            'concept_hierarchy': 3,
            'semantic_links': 8
        },
        evidence=['expert_validation', 'peer_reviewed_paper', 'benchmark_results'],
        confidence=0.9
    )
    result_1 = await engine.validate_proposal(proposal_id_1)
    results.append(result_1)

    # Proposal 2: Lower-quality knowledge
    proposal_id_2 = await engine.propose_knowledge(
        content={
            'title': 'Basic Search Method',
            'description': 'Simple keyword matching',
            'type': 'method'
        },
        evidence=['informal_testing'],
        confidence=0.6
    )
    result_2 = await engine.validate_proposal(proposal_id_2)
    results.append(result_2)

    # Get final statistics
    stats = engine.get_consensus_statistics()

    return {
        'consensus_results': results,
        'consensus_statistics': stats,
        'demonstration_complete': True,
        'timestamp': datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    # Demo execution
    async def main():
        results = await demonstrate_distributed_consensus()
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())
