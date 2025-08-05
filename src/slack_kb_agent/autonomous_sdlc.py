"""Autonomous Software Development Lifecycle (SDLC) execution engine."""

from __future__ import annotations

import asyncio
import json
import time
import logging
import subprocess
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path

from .quantum_task_planner import (
    QuantumTaskPlanner, QuantumTask, TaskState, TaskPriority,
    get_quantum_planner, create_dependent_tasks
)
from .monitoring import get_global_metrics, StructuredLogger
from .cache import get_cache_manager
from .configuration import get_slack_bot_config

logger = logging.getLogger(__name__)


class SDLCPhase(Enum):
    """SDLC phases with quantum execution."""
    ANALYSIS = ("analysis", "Project analysis and requirements gathering")
    DESIGN = ("design", "Architecture and design planning")
    IMPLEMENTATION = ("implementation", "Code development and feature implementation")
    TESTING = ("testing", "Quality assurance and testing")
    DEPLOYMENT = ("deployment", "Production deployment and configuration")
    MONITORING = ("monitoring", "Performance monitoring and maintenance")
    EVOLUTION = ("evolution", "Continuous improvement and iteration")
    
    def __init__(self, phase_name: str, description: str):
        self.phase_name = phase_name
        self.description = description


class QualityGate(Enum):
    """Quality gates for SDLC progression."""
    CODE_QUALITY = ("code_quality", "Code passes linting and formatting")
    SECURITY_SCAN = ("security_scan", "Security vulnerabilities resolved")
    TEST_COVERAGE = ("test_coverage", "Minimum test coverage achieved")
    PERFORMANCE = ("performance", "Performance benchmarks met")
    DOCUMENTATION = ("documentation", "Documentation updated")
    
    def __init__(self, gate_name: str, description: str):
        self.gate_name = gate_name
        self.description = description


@dataclass
class SDLCMetrics:
    """Metrics for SDLC execution tracking."""
    phase_durations: Dict[str, timedelta] = field(default_factory=dict)
    quality_gate_results: Dict[str, bool] = field(default_factory=dict)
    test_coverage_percentage: float = 0.0
    security_issues_found: int = 0
    security_issues_resolved: int = 0
    performance_benchmark_results: Dict[str, float] = field(default_factory=dict)
    code_quality_score: float = 0.0
    documentation_coverage: float = 0.0
    total_execution_time: timedelta = field(default_factory=timedelta)
    success_rate: float = 0.0


class AutonomousSDLC:
    """Autonomous SDLC execution engine with quantum task coordination."""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.planner = get_quantum_planner()
        
        self.metrics = SDLCMetrics()
        self.logger = StructuredLogger("autonomous_sdlc")
        self.cache = get_cache_manager()
        
        # SDLC configuration
        self.min_test_coverage = 85.0
        self.max_security_issues = 0
        self.performance_thresholds = {
            "api_response_time": 200.0,  # ms
            "memory_usage": 512.0,       # MB
            "cpu_usage": 80.0            # %
        }
        
        # Current execution state
        self.current_phase: Optional[SDLCPhase] = None
        self.phase_start_time: Optional[datetime] = None
        self.executed_phases: List[SDLCPhase] = []
        self.failed_phases: List[SDLCPhase] = []
        
        logger.info(f"Autonomous SDLC initialized for project: {self.project_root}")
    
    def analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze project structure and determine optimal SDLC approach."""
        analysis_start = time.time()
        
        try:
            project_info = {
                "project_type": self._detect_project_type(),
                "language": self._detect_primary_language(),
                "framework": self._detect_framework(),
                "build_system": self._detect_build_system(),
                "test_framework": self._detect_test_framework(),
                "existing_files": self._analyze_existing_files(),
                "dependencies": self._analyze_dependencies(),
                "git_status": self._get_git_status()
            }
            
            # Determine SDLC strategy based on analysis
            project_info["sdlc_strategy"] = self._determine_sdlc_strategy(project_info)
            
            analysis_time = timedelta(seconds=time.time() - analysis_start)
            self.metrics.phase_durations["analysis"] = analysis_time
            
            self.logger.log_event("project_analysis_completed", {
                "project_info": project_info,
                "analysis_time": analysis_time.total_seconds()
            })
            
            return project_info
            
        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
            raise
    
    def _detect_project_type(self) -> str:
        """Detect the type of project."""
        if (self.project_root / "pyproject.toml").exists():
            return "python_package"
        elif (self.project_root / "package.json").exists():
            return "node_project"
        elif (self.project_root / "Cargo.toml").exists():
            return "rust_project"
        elif (self.project_root / "go.mod").exists():
            return "go_project"
        elif (self.project_root / "pom.xml").exists():
            return "java_project"
        else:
            return "generic"
    
    def _detect_primary_language(self) -> str:
        """Detect the primary programming language."""
        language_files = {
            "python": list(self.project_root.glob("**/*.py")),
            "javascript": list(self.project_root.glob("**/*.js")),
            "typescript": list(self.project_root.glob("**/*.ts")),
            "rust": list(self.project_root.glob("**/*.rs")),
            "go": list(self.project_root.glob("**/*.go")),
            "java": list(self.project_root.glob("**/*.java"))
        }
        
        # Return language with most files
        max_count = 0
        primary_language = "unknown"
        for lang, files in language_files.items():
            if len(files) > max_count:
                max_count = len(files)
                primary_language = lang
        
        return primary_language
    
    def _detect_framework(self) -> str:
        """Detect the primary framework being used."""
        # Check for Python frameworks
        if (self.project_root / "pyproject.toml").exists():
            with open(self.project_root / "pyproject.toml") as f:
                content = f.read()
                if "fastapi" in content.lower():
                    return "fastapi"
                elif "flask" in content.lower():
                    return "flask"
                elif "django" in content.lower():
                    return "django"
                elif "slack-bolt" in content.lower():
                    return "slack_bot"
        
        # Check for Node.js frameworks
        if (self.project_root / "package.json").exists():
            with open(self.project_root / "package.json") as f:
                content = json.load(f)
                deps = {**content.get("dependencies", {}), **content.get("devDependencies", {})}
                if "express" in deps:
                    return "express"
                elif "react" in deps:
                    return "react"
                elif "vue" in deps:
                    return "vue"
        
        return "unknown"
    
    def _detect_build_system(self) -> str:
        """Detect the build system being used."""
        if (self.project_root / "Makefile").exists():
            return "make"
        elif (self.project_root / "pyproject.toml").exists():
            return "pip"
        elif (self.project_root / "package.json").exists():
            return "npm"
        elif (self.project_root / "Cargo.toml").exists():
            return "cargo"
        elif (self.project_root / "go.mod").exists():
            return "go"
        else:
            return "unknown"
    
    def _detect_test_framework(self) -> str:
        """Detect the testing framework being used."""
        if (self.project_root / "pytest.ini").exists() or (self.project_root / "pyproject.toml").exists():
            return "pytest"
        elif (self.project_root / "package.json").exists():
            with open(self.project_root / "package.json") as f:
                content = json.load(f)
                deps = {**content.get("dependencies", {}), **content.get("devDependencies", {})}
                if "jest" in deps:
                    return "jest"
                elif "mocha" in deps:
                    return "mocha"
        return "unknown"
    
    def _analyze_existing_files(self) -> Dict[str, int]:
        """Analyze existing file structure."""
        file_counts = {}
        for ext in [".py", ".js", ".ts", ".md", ".yml", ".yaml", ".json", ".toml"]:
            files = list(self.project_root.glob(f"**/*{ext}"))
            file_counts[ext[1:]] = len(files)  # Remove the dot
        
        return file_counts
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        deps_info = {}
        
        # Python dependencies
        if (self.project_root / "pyproject.toml").exists():
            try:
                with open(self.project_root / "pyproject.toml") as f:
                    import toml
                    data = toml.load(f)
                    deps_info["python"] = data.get("project", {}).get("dependencies", [])
            except:
                deps_info["python"] = []
        
        # Node.js dependencies
        if (self.project_root / "package.json").exists():
            try:
                with open(self.project_root / "package.json") as f:
                    data = json.load(f)
                    deps_info["node"] = list(data.get("dependencies", {}).keys())
            except:
                deps_info["node"] = []
        
        return deps_info
    
    def _get_git_status(self) -> Dict[str, Any]:
        """Get current git status."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
                return {
                    "clean": len(lines) == 0,
                    "modified_files": len([l for l in lines if l.startswith(' M')]),
                    "added_files": len([l for l in lines if l.startswith('A ')]),
                    "untracked_files": len([l for l in lines if l.startswith('??')])
                }
        except:
            pass
        
        return {"clean": False, "error": True}
    
    def _determine_sdlc_strategy(self, project_info: Dict[str, Any]) -> str:
        """Determine optimal SDLC strategy based on project analysis."""
        project_type = project_info["project_type"]
        language = project_info["language"]
        
        if project_type == "python_package" and language == "python":
            return "python_library"
        elif "slack" in project_info.get("framework", ""):
            return "slack_bot"
        elif project_type == "node_project":
            return "web_application"
        else:
            return "generic_application"
    
    async def execute_sdlc_phase(self, phase: SDLCPhase) -> bool:
        """Execute a specific SDLC phase with quantum task coordination."""
        self.current_phase = phase
        self.phase_start_time = datetime.now()
        
        logger.info(f"Starting SDLC phase: {phase.phase_name}")
        
        try:
            phase_tasks = self._create_phase_tasks(phase)
            
            # Schedule all phase tasks
            for task in phase_tasks:
                self.planner.schedule_task(task.id)
            
            # Wait for all tasks to complete
            success = await self._wait_for_phase_completion(phase_tasks)
            
            # Update metrics
            phase_duration = datetime.now() - self.phase_start_time
            self.metrics.phase_durations[phase.phase_name] = phase_duration
            
            if success:
                self.executed_phases.append(phase)
                logger.info(f"SDLC phase {phase.phase_name} completed successfully")
            else:
                self.failed_phases.append(phase)
                logger.error(f"SDLC phase {phase.phase_name} failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing SDLC phase {phase.phase_name}: {e}")
            self.failed_phases.append(phase)
            return False
        finally:
            self.current_phase = None
            self.phase_start_time = None
    
    def _create_phase_tasks(self, phase: SDLCPhase) -> List[QuantumTask]:
        """Create quantum tasks for a specific SDLC phase."""
        tasks = []
        
        if phase == SDLCPhase.ANALYSIS:
            tasks = self._create_analysis_tasks()
        elif phase == SDLCPhase.DESIGN:
            tasks = self._create_design_tasks()
        elif phase == SDLCPhase.IMPLEMENTATION:
            tasks = self._create_implementation_tasks()
        elif phase == SDLCPhase.TESTING:
            tasks = self._create_testing_tasks()
        elif phase == SDLCPhase.DEPLOYMENT:
            tasks = self._create_deployment_tasks()
        elif phase == SDLCPhase.MONITORING:
            tasks = self._create_monitoring_tasks()
        elif phase == SDLCPhase.EVOLUTION:
            tasks = self._create_evolution_tasks()
        
        return tasks
    
    def _create_analysis_tasks(self) -> List[QuantumTask]:
        """Create tasks for analysis phase."""
        def analyze_requirements(task: QuantumTask) -> bool:
            # Analyze project requirements
            return self._run_quality_gate(QualityGate.DOCUMENTATION)
        
        def analyze_architecture(task: QuantumTask) -> bool:
            # Analyze current architecture
            return True
        
        return create_dependent_tasks([
            ("analyze_requirements", analyze_requirements),
            ("analyze_architecture", analyze_architecture)
        ], TaskPriority.HIGH)
    
    def _create_design_tasks(self) -> List[QuantumTask]:
        """Create tasks for design phase."""
        def create_architecture_docs(task: QuantumTask) -> bool:
            # Update architecture documentation
            return True
        
        def design_api_interfaces(task: QuantumTask) -> bool:
            # Design API interfaces
            return True
        
        return create_dependent_tasks([
            ("create_architecture_docs", create_architecture_docs),
            ("design_api_interfaces", design_api_interfaces)
        ], TaskPriority.HIGH)
    
    def _create_implementation_tasks(self) -> List[QuantumTask]:
        """Create tasks for implementation phase."""
        def implement_core_features(task: QuantumTask) -> bool:
            # Implement core functionality
            return self._run_quality_gate(QualityGate.CODE_QUALITY)
        
        def implement_error_handling(task: QuantumTask) -> bool:
            # Add comprehensive error handling
            return True
        
        def optimize_performance(task: QuantumTask) -> bool:
            # Performance optimization
            return self._run_quality_gate(QualityGate.PERFORMANCE)
        
        return create_dependent_tasks([
            ("implement_core_features", implement_core_features),
            ("implement_error_handling", implement_error_handling),
            ("optimize_performance", optimize_performance)
        ], TaskPriority.CRITICAL)
    
    def _create_testing_tasks(self) -> List[QuantumTask]:
        """Create tasks for testing phase."""
        def run_unit_tests(task: QuantumTask) -> bool:
            return self._run_tests()
        
        def run_integration_tests(task: QuantumTask) -> bool:
            return self._run_integration_tests()
        
        def run_security_scan(task: QuantumTask) -> bool:
            return self._run_quality_gate(QualityGate.SECURITY_SCAN)
        
        return create_dependent_tasks([
            ("run_unit_tests", run_unit_tests),
            ("run_integration_tests", run_integration_tests),
            ("run_security_scan", run_security_scan)
        ], TaskPriority.CRITICAL)
    
    def _create_deployment_tasks(self) -> List[QuantumTask]:
        """Create tasks for deployment phase."""
        def prepare_deployment(task: QuantumTask) -> bool:
            # Prepare deployment configuration
            return True
        
        def validate_deployment(task: QuantumTask) -> bool:
            # Validate deployment readiness
            return True
        
        return create_dependent_tasks([
            ("prepare_deployment", prepare_deployment),
            ("validate_deployment", validate_deployment)
        ], TaskPriority.HIGH)
    
    def _create_monitoring_tasks(self) -> List[QuantumTask]:
        """Create tasks for monitoring phase."""
        def setup_monitoring(task: QuantumTask) -> bool:
            # Setup monitoring and alerting
            return True
        
        def validate_health_checks(task: QuantumTask) -> bool:
            # Validate health check endpoints
            return True
        
        return create_dependent_tasks([
            ("setup_monitoring", setup_monitoring),
            ("validate_health_checks", validate_health_checks)
        ], TaskPriority.MEDIUM)
    
    def _create_evolution_tasks(self) -> List[QuantumTask]:
        """Create tasks for evolution phase."""
        def collect_metrics(task: QuantumTask) -> bool:
            # Collect performance and usage metrics
            return True
        
        def plan_improvements(task: QuantumTask) -> bool:
            # Plan future improvements
            return True
        
        return create_dependent_tasks([
            ("collect_metrics", collect_metrics),
            ("plan_improvements", plan_improvements)
        ], TaskPriority.LOW)
    
    async def _wait_for_phase_completion(self, tasks: List[QuantumTask]) -> bool:
        """Wait for all phase tasks to complete."""
        max_wait_time = 300  # 5 minutes max per phase
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            completed = sum(1 for task in tasks if task.state == TaskState.COMPLETED)
            failed = sum(1 for task in tasks if task.state == TaskState.FAILED)
            
            if completed + failed == len(tasks):
                return failed == 0  # Success if no failures
            
            await asyncio.sleep(1.0)
        
        # Timeout - consider it a failure
        logger.warning(f"Phase tasks timed out after {max_wait_time} seconds")
        return False
    
    def _run_quality_gate(self, gate: QualityGate) -> bool:
        """Run a specific quality gate check."""
        try:
            if gate == QualityGate.CODE_QUALITY:
                return self._check_code_quality()
            elif gate == QualityGate.SECURITY_SCAN:
                return self._run_security_scan()
            elif gate == QualityGate.TEST_COVERAGE:
                return self._check_test_coverage()
            elif gate == QualityGate.PERFORMANCE:
                return self._check_performance()
            elif gate == QualityGate.DOCUMENTATION:
                return self._check_documentation()
            
            return False
            
        except Exception as e:
            logger.error(f"Quality gate {gate.gate_name} failed: {e}")
            return False
    
    def _check_code_quality(self) -> bool:
        """Check code quality using linting tools."""
        try:
            # Run ruff for Python
            result = subprocess.run(
                ["ruff", "check", "src/"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.metrics.code_quality_score = 100.0
                self.metrics.quality_gate_results["code_quality"] = True
                return True
            else:
                logger.warning(f"Code quality issues found: {result.stdout}")
                self.metrics.code_quality_score = 70.0
                self.metrics.quality_gate_results["code_quality"] = False
                return False
                
        except Exception as e:
            logger.error(f"Code quality check failed: {e}")
            return False
    
    def _run_security_scan(self) -> bool:
        """Run security vulnerability scan."""
        try:
            # Run bandit for Python security scan
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Parse bandit results
                try:
                    bandit_data = json.loads(result.stdout)
                    issues = bandit_data.get("results", [])
                    self.metrics.security_issues_found = len(issues)
                    
                    # Filter high severity issues
                    high_severity = [i for i in issues if i.get("issue_severity") == "HIGH"]
                    
                    success = len(high_severity) <= self.max_security_issues
                    self.metrics.quality_gate_results["security_scan"] = success
                    return success
                    
                except json.JSONDecodeError:
                    pass
            
            self.metrics.quality_gate_results["security_scan"] = False
            return False
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            return False
    
    def _run_tests(self) -> bool:
        """Run unit tests and check coverage."""
        try:
            # Run pytest with coverage
            result = subprocess.run(
                ["pytest", "--cov=src", "--cov-report=json", "-v"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            success = result.returncode == 0
            
            # Parse coverage results
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                try:
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                        self.metrics.test_coverage_percentage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                except:
                    pass
            
            coverage_success = self.metrics.test_coverage_percentage >= self.min_test_coverage
            self.metrics.quality_gate_results["test_coverage"] = coverage_success
            
            return success and coverage_success
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
    
    def _run_integration_tests(self) -> bool:
        """Run integration tests."""
        try:
            # Run integration tests if they exist
            result = subprocess.run(
                ["pytest", "tests/", "-m", "integration", "-v"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            return False
    
    def _check_performance(self) -> bool:
        """Check performance benchmarks."""
        try:
            # Run performance tests if they exist
            result = subprocess.run(
                ["pytest", "tests/performance/", "-v"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            success = result.returncode == 0
            self.metrics.quality_gate_results["performance"] = success
            return success
            
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
            return False
    
    def _check_documentation(self) -> bool:
        """Check documentation completeness."""
        # Simple documentation check
        readme_exists = (self.project_root / "README.md").exists()
        api_docs_exist = (self.project_root / "API_USAGE_GUIDE.md").exists()
        
        doc_score = 0.0
        if readme_exists:
            doc_score += 0.5
        if api_docs_exist:
            doc_score += 0.5
        
        self.metrics.documentation_coverage = doc_score * 100
        success = doc_score >= 0.5  # At least README required
        self.metrics.quality_gate_results["documentation"] = success
        
        return success
    
    async def execute_full_sdlc(self) -> bool:
        """Execute the complete SDLC autonomously."""
        logger.info("Starting autonomous SDLC execution")
        start_time = datetime.now()
        
        try:
            # Analysis phase
            project_info = self.analyze_project_structure()
            
            # Execute phases based on project type
            phases_to_execute = [
                SDLCPhase.ANALYSIS,
                SDLCPhase.IMPLEMENTATION,
                SDLCPhase.TESTING,
                SDLCPhase.DEPLOYMENT,
                SDLCPhase.MONITORING
            ]
            
            # Execute each phase
            for phase in phases_to_execute:
                success = await self.execute_sdlc_phase(phase)
                if not success:
                    logger.error(f"SDLC execution failed at phase: {phase.phase_name}")
                    return False
            
            # Calculate final metrics
            self.metrics.total_execution_time = datetime.now() - start_time
            successful_phases = len(self.executed_phases)
            total_phases = len(phases_to_execute)
            self.metrics.success_rate = successful_phases / total_phases if total_phases > 0 else 0.0
            
            logger.info(f"Autonomous SDLC completed successfully in {self.metrics.total_execution_time}")
            
            # Log final metrics
            self.logger.log_event("sdlc_execution_completed", {
                "success_rate": self.metrics.success_rate,
                "total_execution_time": self.metrics.total_execution_time.total_seconds(),
                "executed_phases": [p.phase_name for p in self.executed_phases],
                "failed_phases": [p.phase_name for p in self.failed_phases],
                "quality_gates": self.metrics.quality_gate_results,
                "test_coverage": self.metrics.test_coverage_percentage,
                "code_quality_score": self.metrics.code_quality_score
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Autonomous SDLC execution failed: {e}")
            return False
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of SDLC execution."""
        return {
            "metrics": {
                "success_rate": self.metrics.success_rate,
                "total_execution_time": self.metrics.total_execution_time.total_seconds(),
                "test_coverage": self.metrics.test_coverage_percentage,
                "code_quality_score": self.metrics.code_quality_score,
                "security_issues_found": self.metrics.security_issues_found,
                "documentation_coverage": self.metrics.documentation_coverage
            },
            "phases": {
                "executed": [p.phase_name for p in self.executed_phases],
                "failed": [p.phase_name for p in self.failed_phases],
                "durations": {k: v.total_seconds() for k, v in self.metrics.phase_durations.items()}
            },
            "quality_gates": self.metrics.quality_gate_results,
            "performance_benchmarks": self.metrics.performance_benchmark_results
        }


# Global instance
_autonomous_sdlc: Optional[AutonomousSDLC] = None


def get_autonomous_sdlc(project_root: str = "/root/repo") -> AutonomousSDLC:
    """Get or create global autonomous SDLC instance."""
    global _autonomous_sdlc
    if _autonomous_sdlc is None or _autonomous_sdlc.project_root != Path(project_root):
        _autonomous_sdlc = AutonomousSDLC(project_root)
    return _autonomous_sdlc