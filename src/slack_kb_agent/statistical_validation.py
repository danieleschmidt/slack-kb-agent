"""Comprehensive statistical validation and benchmarking framework for research algorithms.

This module provides rigorous statistical validation, significance testing, and 
benchmarking capabilities to ensure research algorithms meet publication standards
and demonstrate statistically significant improvements over baselines.
"""

import asyncio
import json
import time
import logging
import math
import numpy as np
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from enum import Enum

try:
    from scipy import stats
    from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, ks_2samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    ttest_ind = None
    wilcoxon = None
    mannwhitneyu = None
    ks_2samp = None

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of statistical tests."""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


class EffectSize(Enum):
    """Effect size categories."""
    NEGLIGIBLE = "negligible"  # < 0.2
    SMALL = "small"           # 0.2 - 0.5
    MEDIUM = "medium"         # 0.5 - 0.8
    LARGE = "large"           # > 0.8


@dataclass
class StatisticalTest:
    """Statistical test configuration and results."""
    test_type: TestType
    test_name: str
    description: str
    assumptions: List[str]
    alpha_level: float = 0.05
    two_tailed: bool = True
    min_sample_size: int = 10


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    algorithm_name: str
    test_name: str
    metric_name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    algorithm_name: str
    baseline_algorithm: str
    test_results: Dict[str, Any]
    effect_sizes: Dict[str, float]
    statistical_significance: Dict[str, bool]
    confidence_intervals: Dict[str, Tuple[float, float]]
    power_analysis: Dict[str, float]
    practical_significance: bool
    publication_ready: bool
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)


class StatisticalValidator:
    """Comprehensive statistical validation framework."""
    
    def __init__(self, alpha_level: float = 0.05, power_threshold: float = 0.8):
        self.alpha_level = alpha_level
        self.power_threshold = power_threshold
        
        # Statistical tests registry
        self.available_tests = {
            TestType.T_TEST: StatisticalTest(
                test_type=TestType.T_TEST,
                test_name="Student's t-test",
                description="Parametric test for comparing means of two groups",
                assumptions=["Normal distribution", "Equal variances", "Independent samples"],
                min_sample_size=30
            ),
            TestType.WILCOXON: StatisticalTest(
                test_type=TestType.WILCOXON,
                test_name="Wilcoxon signed-rank test",
                description="Non-parametric test for paired samples",
                assumptions=["Paired samples", "Ordinal data"],
                min_sample_size=10
            ),
            TestType.MANN_WHITNEY: StatisticalTest(
                test_type=TestType.MANN_WHITNEY,
                test_name="Mann-Whitney U test",
                description="Non-parametric test for independent samples",
                assumptions=["Independent samples", "Ordinal data"],
                min_sample_size=10
            ),
            TestType.BOOTSTRAP: StatisticalTest(
                test_type=TestType.BOOTSTRAP,
                test_name="Bootstrap resampling test",
                description="Distribution-free resampling method",
                assumptions=["Representative sample"],
                min_sample_size=20
            )
        }
        
        # Validation history
        self.validation_history: deque = deque(maxlen=1000)
        self.benchmark_results: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        
        logger.info("Statistical Validator initialized")
    
    async def validate_algorithm_performance(self, 
                                           algorithm_results: List[float],
                                           baseline_results: List[float],
                                           algorithm_name: str,
                                           baseline_name: str = "baseline",
                                           metric_name: str = "accuracy") -> ValidationReport:
        """Comprehensive validation of algorithm performance against baseline."""
        try:
            start_time = datetime.utcnow()
            
            # Input validation
            if len(algorithm_results) < 10 or len(baseline_results) < 10:
                raise ValueError("Insufficient data for statistical validation (minimum 10 samples)")
            
            # Store results
            await self._store_benchmark_results(algorithm_name, algorithm_results, metric_name)
            await self._store_benchmark_results(baseline_name, baseline_results, metric_name)
            
            # Perform comprehensive statistical tests
            test_results = await self._perform_statistical_tests(
                algorithm_results, baseline_results, metric_name
            )
            
            # Calculate effect sizes
            effect_sizes = await self._calculate_effect_sizes(
                algorithm_results, baseline_results
            )
            
            # Power analysis
            power_analysis = await self._perform_power_analysis(
                algorithm_results, baseline_results
            )
            
            # Confidence intervals
            confidence_intervals = await self._calculate_confidence_intervals(
                algorithm_results, baseline_results
            )
            
            # Determine statistical significance
            statistical_significance = {}
            for test_name, result in test_results.items():
                statistical_significance[test_name] = result.get('p_value', 1.0) < self.alpha_level
            
            # Assess practical significance
            practical_significance = await self._assess_practical_significance(
                algorithm_results, baseline_results, effect_sizes
            )
            
            # Publication readiness assessment
            publication_ready = await self._assess_publication_readiness(
                test_results, effect_sizes, power_analysis, statistical_significance
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                test_results, effect_sizes, power_analysis, publication_ready
            )
            
            # Create validation report
            report = ValidationReport(
                algorithm_name=algorithm_name,
                baseline_algorithm=baseline_name,
                test_results=test_results,
                effect_sizes=effect_sizes,
                statistical_significance=statistical_significance,
                confidence_intervals=confidence_intervals,
                power_analysis=power_analysis,
                practical_significance=practical_significance,
                publication_ready=publication_ready,
                recommendations=recommendations
            )
            
            self.validation_history.append(report)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Validation completed for {algorithm_name} vs {baseline_name} in {duration:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Error validating algorithm performance: {e}")
            raise
    
    async def _perform_statistical_tests(self, 
                                       algorithm_results: List[float],
                                       baseline_results: List[float],
                                       metric_name: str) -> Dict[str, Any]:
        """Perform comprehensive statistical tests."""
        test_results = {}
        
        try:
            # Check data distribution and choose appropriate tests
            normality_algo = await self._test_normality(algorithm_results)
            normality_base = await self._test_normality(baseline_results)
            
            # Parametric tests if data is normally distributed
            if normality_algo and normality_base and SCIPY_AVAILABLE:
                # Student's t-test
                t_stat, p_value = ttest_ind(algorithm_results, baseline_results)
                test_results['t_test'] = {
                    'test_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha_level,
                    'test_type': 'parametric',
                    'assumptions_met': True
                }
            
            # Non-parametric tests (always applicable)
            if SCIPY_AVAILABLE:
                # Mann-Whitney U test
                u_stat, p_value_mw = mannwhitneyu(
                    algorithm_results, baseline_results, alternative='two-sided'
                )
                test_results['mann_whitney'] = {
                    'test_statistic': u_stat,
                    'p_value': p_value_mw,
                    'significant': p_value_mw < self.alpha_level,
                    'test_type': 'non_parametric',
                    'assumptions_met': True
                }
                
                # Kolmogorov-Smirnov test
                ks_stat, p_value_ks = ks_2samp(algorithm_results, baseline_results)
                test_results['kolmogorov_smirnov'] = {
                    'test_statistic': ks_stat,
                    'p_value': p_value_ks,
                    'significant': p_value_ks < self.alpha_level,
                    'test_type': 'non_parametric',
                    'assumptions_met': True
                }
            
            # Bootstrap test (distribution-free)
            bootstrap_result = await self._bootstrap_test(algorithm_results, baseline_results)
            test_results['bootstrap'] = bootstrap_result
            
            # Permutation test
            permutation_result = await self._permutation_test(algorithm_results, baseline_results)
            test_results['permutation'] = permutation_result
            
        except Exception as e:
            logger.error(f"Error performing statistical tests: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    async def _calculate_effect_sizes(self, 
                                    algorithm_results: List[float],
                                    baseline_results: List[float]) -> Dict[str, float]:
        """Calculate multiple effect size measures."""
        effect_sizes = {}
        
        try:
            mean_algo = statistics.mean(algorithm_results)
            mean_base = statistics.mean(baseline_results)
            
            # Cohen's d
            pooled_std = await self._calculate_pooled_std(algorithm_results, baseline_results)
            if pooled_std > 0:
                cohens_d = (mean_algo - mean_base) / pooled_std
                effect_sizes['cohens_d'] = cohens_d
                effect_sizes['cohens_d_interpretation'] = self._interpret_cohens_d(cohens_d)
            
            # Glass's delta
            std_base = statistics.stdev(baseline_results) if len(baseline_results) > 1 else 1.0
            if std_base > 0:
                glass_delta = (mean_algo - mean_base) / std_base
                effect_sizes['glass_delta'] = glass_delta
            
            # Hedges' g (bias-corrected Cohen's d)
            if pooled_std > 0 and len(algorithm_results) > 1 and len(baseline_results) > 1:
                n1, n2 = len(algorithm_results), len(baseline_results)
                correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
                hedges_g = cohens_d * correction_factor
                effect_sizes['hedges_g'] = hedges_g
            
            # Cliff's delta (non-parametric effect size)
            cliffs_delta = await self._calculate_cliffs_delta(algorithm_results, baseline_results)
            effect_sizes['cliffs_delta'] = cliffs_delta
            effect_sizes['cliffs_delta_interpretation'] = self._interpret_cliffs_delta(cliffs_delta)
            
            # Common language effect size
            cles = await self._calculate_common_language_effect_size(algorithm_results, baseline_results)
            effect_sizes['common_language_effect_size'] = cles
            
        except Exception as e:
            logger.error(f"Error calculating effect sizes: {e}")
        
        return effect_sizes
    
    async def _perform_power_analysis(self, 
                                    algorithm_results: List[float],
                                    baseline_results: List[float]) -> Dict[str, float]:
        """Perform statistical power analysis."""
        power_analysis = {}
        
        try:
            n1, n2 = len(algorithm_results), len(baseline_results)
            
            # Calculate observed effect size
            mean_diff = statistics.mean(algorithm_results) - statistics.mean(baseline_results)
            pooled_std = await self._calculate_pooled_std(algorithm_results, baseline_results)
            
            if pooled_std > 0:
                observed_effect_size = abs(mean_diff / pooled_std)
                
                # Estimate statistical power for current sample size
                power = await self._estimate_power(observed_effect_size, n1, n2, self.alpha_level)
                power_analysis['observed_power'] = power
                
                # Calculate required sample size for different power levels
                for target_power in [0.8, 0.9, 0.95]:
                    required_n = await self._calculate_required_sample_size(
                        observed_effect_size, target_power, self.alpha_level
                    )
                    power_analysis[f'required_n_for_power_{target_power}'] = required_n
                
                # Power for different effect sizes
                for effect_size in [0.2, 0.5, 0.8]:  # Small, medium, large
                    power_for_effect = await self._estimate_power(effect_size, n1, n2, self.alpha_level)
                    power_analysis[f'power_for_effect_{effect_size}'] = power_for_effect
                
                power_analysis['effect_size_used'] = observed_effect_size
                power_analysis['sample_sizes'] = {'algorithm': n1, 'baseline': n2}
            
        except Exception as e:
            logger.error(f"Error performing power analysis: {e}")
        
        return power_analysis
    
    async def _calculate_confidence_intervals(self, 
                                            algorithm_results: List[float],
                                            baseline_results: List[float]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for various statistics."""
        confidence_intervals = {}
        
        try:
            confidence_level = 1 - self.alpha_level
            
            # Confidence interval for mean difference
            mean_diff = statistics.mean(algorithm_results) - statistics.mean(baseline_results)
            se_diff = await self._calculate_standard_error_difference(algorithm_results, baseline_results)
            
            if SCIPY_AVAILABLE and se_diff > 0:
                df = len(algorithm_results) + len(baseline_results) - 2
                t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
                margin_of_error = t_critical * se_diff
                
                ci_lower = mean_diff - margin_of_error
                ci_upper = mean_diff + margin_of_error
                confidence_intervals['mean_difference'] = (ci_lower, ci_upper)
            
            # Bootstrap confidence intervals
            bootstrap_ci = await self._bootstrap_confidence_interval(
                algorithm_results, baseline_results, confidence_level
            )
            confidence_intervals['bootstrap_mean_difference'] = bootstrap_ci
            
            # Confidence intervals for individual group means
            ci_algo = await self._calculate_mean_confidence_interval(algorithm_results, confidence_level)
            ci_base = await self._calculate_mean_confidence_interval(baseline_results, confidence_level)
            
            confidence_intervals['algorithm_mean'] = ci_algo
            confidence_intervals['baseline_mean'] = ci_base
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
        
        return confidence_intervals
    
    async def _test_normality(self, data: List[float]) -> bool:
        """Test if data follows normal distribution."""
        if not SCIPY_AVAILABLE or len(data) < 8:
            return False
        
        try:
            # Shapiro-Wilk test for normality
            stat, p_value = stats.shapiro(data)
            return p_value > self.alpha_level  # Null hypothesis: data is normal
        except Exception:
            return False
    
    async def _bootstrap_test(self, 
                            algorithm_results: List[float],
                            baseline_results: List[float],
                            n_bootstrap: int = 10000) -> Dict[str, Any]:
        """Perform bootstrap hypothesis test."""
        try:
            observed_diff = statistics.mean(algorithm_results) - statistics.mean(baseline_results)
            
            # Create combined pool for resampling under null hypothesis
            combined_data = algorithm_results + baseline_results
            n1, n2 = len(algorithm_results), len(baseline_results)
            
            # Bootstrap resampling
            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                # Resample without replacement
                resampled = np.random.choice(combined_data, size=n1+n2, replace=True)
                sample1 = resampled[:n1]
                sample2 = resampled[n1:]
                
                bootstrap_diff = np.mean(sample1) - np.mean(sample2)
                bootstrap_diffs.append(bootstrap_diff)
            
            # Calculate p-value
            extreme_values = sum(1 for diff in bootstrap_diffs if abs(diff) >= abs(observed_diff))
            p_value = extreme_values / n_bootstrap
            
            return {
                'test_statistic': observed_diff,
                'p_value': p_value,
                'significant': p_value < self.alpha_level,
                'test_type': 'bootstrap',
                'n_bootstrap': n_bootstrap,
                'bootstrap_distribution': {
                    'mean': statistics.mean(bootstrap_diffs),
                    'std': statistics.stdev(bootstrap_diffs) if len(bootstrap_diffs) > 1 else 0,
                    'percentiles': {
                        '2.5': np.percentile(bootstrap_diffs, 2.5),
                        '97.5': np.percentile(bootstrap_diffs, 97.5)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in bootstrap test: {e}")
            return {'error': str(e)}
    
    async def _permutation_test(self, 
                              algorithm_results: List[float],
                              baseline_results: List[float],
                              n_permutations: int = 10000) -> Dict[str, Any]:
        """Perform permutation hypothesis test."""
        try:
            observed_diff = statistics.mean(algorithm_results) - statistics.mean(baseline_results)
            
            # Create combined data
            combined_data = algorithm_results + baseline_results
            n1 = len(algorithm_results)
            
            # Permutation test
            permutation_diffs = []
            for _ in range(n_permutations):
                # Randomly permute the combined data
                permuted = np.random.permutation(combined_data)
                group1 = permuted[:n1]
                group2 = permuted[n1:]
                
                perm_diff = np.mean(group1) - np.mean(group2)
                permutation_diffs.append(perm_diff)
            
            # Calculate p-value
            extreme_values = sum(1 for diff in permutation_diffs if abs(diff) >= abs(observed_diff))
            p_value = extreme_values / n_permutations
            
            return {
                'test_statistic': observed_diff,
                'p_value': p_value,
                'significant': p_value < self.alpha_level,
                'test_type': 'permutation',
                'n_permutations': n_permutations
            }
            
        except Exception as e:
            logger.error(f"Error in permutation test: {e}")
            return {'error': str(e)}
    
    async def _calculate_pooled_std(self, group1: List[float], group2: List[float]) -> float:
        """Calculate pooled standard deviation."""
        try:
            n1, n2 = len(group1), len(group2)
            
            if n1 <= 1 or n2 <= 1:
                return 1.0  # Default to avoid division by zero
            
            var1 = statistics.variance(group1)
            var2 = statistics.variance(group2)
            
            pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            return math.sqrt(pooled_variance)
            
        except Exception:
            return 1.0
    
    async def _calculate_cliffs_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        try:
            n1, n2 = len(group1), len(group2)
            
            # Count pairs where group1 > group2, group1 < group2
            greater = 0
            less = 0
            
            for x in group1:
                for y in group2:
                    if x > y:
                        greater += 1
                    elif x < y:
                        less += 1
            
            total_pairs = n1 * n2
            cliffs_delta = (greater - less) / total_pairs if total_pairs > 0 else 0.0
            
            return cliffs_delta
            
        except Exception:
            return 0.0
    
    async def _calculate_common_language_effect_size(self, 
                                                   group1: List[float], 
                                                   group2: List[float]) -> float:
        """Calculate Common Language Effect Size."""
        try:
            total_comparisons = len(group1) * len(group2)
            
            if total_comparisons == 0:
                return 0.5
            
            favorable = 0
            for x in group1:
                for y in group2:
                    if x > y:
                        favorable += 1
                    elif x == y:
                        favorable += 0.5  # Ties count as 0.5
            
            return favorable / total_comparisons
            
        except Exception:
            return 0.5
    
    async def _estimate_power(self, effect_size: float, n1: int, n2: int, alpha: float) -> float:
        """Estimate statistical power for given parameters."""
        try:
            if not SCIPY_AVAILABLE:
                # Simplified power estimation
                return min(1.0, max(0.0, (effect_size * math.sqrt(min(n1, n2)) / 3)))
            
            # More accurate power calculation using scipy
            # This is a simplified approximation
            df = n1 + n2 - 2
            t_critical = stats.t.ppf(1 - alpha/2, df)
            
            # Standard error for two-sample test
            se = math.sqrt(1/n1 + 1/n2)
            ncp = effect_size / se  # Non-centrality parameter
            
            # Power is P(|T| > t_critical | δ ≠ 0)
            power = 1 - stats.t.cdf(t_critical, df, loc=ncp) + stats.t.cdf(-t_critical, df, loc=ncp)
            
            return min(1.0, max(0.0, power))
            
        except Exception:
            return 0.5
    
    async def _calculate_required_sample_size(self, 
                                            effect_size: float, 
                                            power: float, 
                                            alpha: float) -> int:
        """Calculate required sample size for desired power."""
        try:
            if effect_size == 0:
                return float('inf')
            
            # Simplified sample size calculation
            # More sophisticated methods would use iterative approaches
            z_alpha = 1.96 if alpha == 0.05 else stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power) if SCIPY_AVAILABLE else 0.84  # For power = 0.8
            
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            
            return max(10, int(math.ceil(n)))
            
        except Exception:
            return 100  # Default reasonable sample size
    
    async def _calculate_standard_error_difference(self, 
                                                 group1: List[float], 
                                                 group2: List[float]) -> float:
        """Calculate standard error of the difference between means."""
        try:
            n1, n2 = len(group1), len(group2)
            
            if n1 <= 1 or n2 <= 1:
                return 1.0
            
            var1 = statistics.variance(group1)
            var2 = statistics.variance(group2)
            
            se_diff = math.sqrt(var1/n1 + var2/n2)
            return se_diff
            
        except Exception:
            return 1.0
    
    async def _bootstrap_confidence_interval(self, 
                                           group1: List[float], 
                                           group2: List[float],
                                           confidence_level: float,
                                           n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for mean difference."""
        try:
            bootstrap_diffs = []
            
            for _ in range(n_bootstrap):
                # Bootstrap resample each group
                sample1 = np.random.choice(group1, size=len(group1), replace=True)
                sample2 = np.random.choice(group2, size=len(group2), replace=True)
                
                diff = np.mean(sample1) - np.mean(sample2)
                bootstrap_diffs.append(diff)
            
            # Calculate percentiles for confidence interval
            alpha = 1 - confidence_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
            ci_upper = np.percentile(bootstrap_diffs, upper_percentile)
            
            return (ci_lower, ci_upper)
            
        except Exception:
            return (0.0, 0.0)
    
    async def _calculate_mean_confidence_interval(self, 
                                                data: List[float], 
                                                confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for a single mean."""
        try:
            n = len(data)
            if n <= 1:
                return (0.0, 0.0)
            
            mean = statistics.mean(data)
            sem = statistics.stdev(data) / math.sqrt(n)
            
            if SCIPY_AVAILABLE:
                df = n - 1
                t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
            else:
                t_critical = 1.96  # Approximation for large samples
            
            margin_of_error = t_critical * sem
            
            return (mean - margin_of_error, mean + margin_of_error)
            
        except Exception:
            return (0.0, 0.0)
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_cliffs_delta(self, cliffs_delta: float) -> str:
        """Interpret Cliff's delta effect size."""
        abs_delta = abs(cliffs_delta)
        if abs_delta < 0.11:
            return "negligible"
        elif abs_delta < 0.28:
            return "small"
        elif abs_delta < 0.43:
            return "medium"
        else:
            return "large"
    
    async def _assess_practical_significance(self, 
                                           algorithm_results: List[float],
                                           baseline_results: List[float],
                                           effect_sizes: Dict[str, float]) -> bool:
        """Assess practical significance of the results."""
        try:
            # Multiple criteria for practical significance
            criteria_met = 0
            total_criteria = 0
            
            # Criterion 1: Effect size is at least small
            if 'cohens_d' in effect_sizes:
                total_criteria += 1
                if abs(effect_sizes['cohens_d']) >= 0.2:
                    criteria_met += 1
            
            # Criterion 2: Mean improvement is meaningful (> 5%)
            mean_algo = statistics.mean(algorithm_results)
            mean_base = statistics.mean(baseline_results)
            
            if mean_base != 0:
                total_criteria += 1
                improvement = (mean_algo - mean_base) / abs(mean_base)
                if improvement > 0.05:  # 5% improvement threshold
                    criteria_met += 1
            
            # Criterion 3: Common language effect size > 0.6
            if 'common_language_effect_size' in effect_sizes:
                total_criteria += 1
                if effect_sizes['common_language_effect_size'] > 0.6:
                    criteria_met += 1
            
            # Practical significance if majority of criteria are met
            return criteria_met / total_criteria >= 0.67 if total_criteria > 0 else False
            
        except Exception:
            return False
    
    async def _assess_publication_readiness(self, 
                                          test_results: Dict[str, Any],
                                          effect_sizes: Dict[str, float],
                                          power_analysis: Dict[str, float],
                                          statistical_significance: Dict[str, bool]) -> bool:
        """Assess if results are ready for publication."""
        try:
            readiness_score = 0
            max_score = 0
            
            # Criterion 1: Statistical significance in multiple tests
            significant_tests = sum(1 for sig in statistical_significance.values() if sig)
            max_score += 2
            if significant_tests >= 2:
                readiness_score += 2
            elif significant_tests >= 1:
                readiness_score += 1
            
            # Criterion 2: Adequate statistical power
            if 'observed_power' in power_analysis:
                max_score += 2
                power = power_analysis['observed_power']
                if power >= 0.9:
                    readiness_score += 2
                elif power >= 0.8:
                    readiness_score += 1
            
            # Criterion 3: Meaningful effect size
            if 'cohens_d' in effect_sizes:
                max_score += 2
                effect_interpretation = self._interpret_cohens_d(effect_sizes['cohens_d'])
                if effect_interpretation in ['large', 'medium']:
                    readiness_score += 2
                elif effect_interpretation == 'small':
                    readiness_score += 1
            
            # Criterion 4: Multiple effect size measures agree
            if len(effect_sizes) >= 2:
                max_score += 1
                # Check if effect sizes are consistent
                effect_values = [v for v in effect_sizes.values() if isinstance(v, (int, float))]
                if effect_values and all(abs(v) >= 0.2 for v in effect_values):
                    readiness_score += 1
            
            # Publication ready if score is above threshold
            return readiness_score / max_score >= 0.75 if max_score > 0 else False
            
        except Exception:
            return False
    
    async def _generate_recommendations(self, 
                                      test_results: Dict[str, Any],
                                      effect_sizes: Dict[str, float],
                                      power_analysis: Dict[str, float],
                                      publication_ready: bool) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        try:
            # Power-based recommendations
            if 'observed_power' in power_analysis:
                power = power_analysis['observed_power']
                if power < 0.8:
                    if 'required_n_for_power_0.8' in power_analysis:
                        required_n = power_analysis['required_n_for_power_0.8']
                        recommendations.append(
                            f"Increase sample size to {required_n} per group for adequate power (0.8)"
                        )
                    else:
                        recommendations.append("Increase sample size for adequate statistical power")
            
            # Effect size recommendations
            if 'cohens_d' in effect_sizes:
                effect_interpretation = self._interpret_cohens_d(effect_sizes['cohens_d'])
                if effect_interpretation == 'negligible':
                    recommendations.append(
                        "Effect size is negligible - consider algorithm improvements or different metrics"
                    )
                elif effect_interpretation == 'small':
                    recommendations.append(
                        "Effect size is small but meaningful - consider collecting more data"
                    )
            
            # Statistical significance recommendations
            significant_count = sum(1 for result in test_results.values() 
                                  if isinstance(result, dict) and result.get('significant', False))
            
            if significant_count == 0:
                recommendations.append(
                    "No statistically significant results found - review algorithm or experimental design"
                )
            elif significant_count == 1:
                recommendations.append(
                    "Only one test shows significance - validate with additional tests or larger sample"
                )
            
            # Publication readiness
            if not publication_ready:
                recommendations.append(
                    "Results not yet publication-ready - address power, effect size, or significance issues"
                )
            else:
                recommendations.append(
                    "Results meet publication standards - ready for academic submission"
                )
            
            # Data quality recommendations
            if not test_results.get('error'):
                recommendations.append("Consider replicating results with independent datasets")
                recommendations.append("Perform sensitivity analysis with different parameters")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    async def _store_benchmark_results(self, 
                                     algorithm_name: str, 
                                     results: List[float], 
                                     metric_name: str) -> None:
        """Store benchmark results for historical tracking."""
        try:
            for value in results:
                result = BenchmarkResult(
                    algorithm_name=algorithm_name,
                    test_name="validation_benchmark",
                    metric_name=metric_name,
                    value=value,
                    timestamp=datetime.utcnow()
                )
                self.benchmark_results[algorithm_name].append(result)
                
        except Exception as e:
            logger.error(f"Error storing benchmark results: {e}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        try:
            if not self.validation_history:
                return {'total_validations': 0}
            
            total_validations = len(self.validation_history)
            publication_ready = sum(1 for report in self.validation_history if report.publication_ready)
            practical_significant = sum(1 for report in self.validation_history if report.practical_significance)
            
            # Average effect sizes
            all_cohens_d = [report.effect_sizes.get('cohens_d', 0) 
                           for report in self.validation_history 
                           if 'cohens_d' in report.effect_sizes]
            
            avg_effect_size = statistics.mean(all_cohens_d) if all_cohens_d else 0.0
            
            return {
                'total_validations': total_validations,
                'publication_ready_rate': publication_ready / total_validations,
                'practical_significance_rate': practical_significant / total_validations,
                'average_effect_size': avg_effect_size,
                'algorithms_tested': len(self.benchmark_results),
                'total_benchmark_results': sum(len(results) for results in self.benchmark_results.values()),
                'scipy_available': SCIPY_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"Error calculating validation statistics: {e}")
            return {'error': str(e)}


# Global validator instance
_statistical_validator_instance: Optional[StatisticalValidator] = None


def get_statistical_validator() -> StatisticalValidator:
    """Get or create global statistical validator instance."""
    global _statistical_validator_instance
    if _statistical_validator_instance is None:
        _statistical_validator_instance = StatisticalValidator()
    return _statistical_validator_instance


async def demonstrate_statistical_validation() -> Dict[str, Any]:
    """Demonstrate statistical validation capabilities."""
    validator = get_statistical_validator()
    
    # Generate realistic algorithm performance data
    np.random.seed(42)  # For reproducible results
    
    # Algorithm results (better performance)
    algorithm_results = list(np.random.normal(0.85, 0.08, 50))  # Mean=0.85, std=0.08
    algorithm_results = [max(0, min(1, x)) for x in algorithm_results]  # Clamp to [0,1]
    
    # Baseline results (worse performance)
    baseline_results = list(np.random.normal(0.78, 0.09, 50))   # Mean=0.78, std=0.09
    baseline_results = [max(0, min(1, x)) for x in baseline_results]    # Clamp to [0,1]
    
    # Perform comprehensive validation
    report = await validator.validate_algorithm_performance(
        algorithm_results=algorithm_results,
        baseline_results=baseline_results,
        algorithm_name="QuantumEnhancedSearch",
        baseline_name="StandardSearch",
        metric_name="accuracy"
    )
    
    # Get validation statistics
    stats = validator.get_validation_statistics()
    
    return {
        'validation_report': report,
        'validation_statistics': stats,
        'demonstration_complete': True,
        'timestamp': datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    # Demo execution
    async def main():
        results = await demonstrate_statistical_validation()
        print(json.dumps(results, indent=2, default=str))
    
    asyncio.run(main())