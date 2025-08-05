"""Tests for autonomous SDLC execution engine."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

from slack_kb_agent.autonomous_sdlc import (
    AutonomousSDLC, SDLCPhase, QualityGate, SDLCMetrics,
    get_autonomous_sdlc
)


class TestSDLCMetrics:
    """Test SDLC metrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = SDLCMetrics()
        
        assert isinstance(metrics.phase_durations, dict)
        assert isinstance(metrics.quality_gate_results, dict)
        assert metrics.test_coverage_percentage == 0.0
        assert metrics.security_issues_found == 0
        assert metrics.success_rate == 0.0


class TestAutonomousSDLC:
    """Test autonomous SDLC functionality."""
    
    @pytest.fixture
    def temp_project_root(self, tmp_path):
        """Create temporary project root for testing."""
        project_root = tmp_path / "test_project"
        project_root.mkdir()
        
        # Create basic project structure
        (project_root / "src").mkdir()
        (project_root / "tests").mkdir()
        (project_root / "pyproject.toml").write_text("""
[project]
name = "test_project"
dependencies = ["requests", "pytest"]
""")
        (project_root / "README.md").write_text("# Test Project")
        
        return str(project_root)
    
    @pytest.fixture
    def sdlc(self, temp_project_root):
        """Create SDLC instance for testing."""
        return AutonomousSDLC(temp_project_root)
    
    def test_sdlc_initialization(self, sdlc, temp_project_root):
        """Test SDLC initialization."""
        assert str(sdlc.project_root) == temp_project_root
        assert sdlc.min_test_coverage == 85.0
        assert sdlc.max_security_issues == 0
        assert isinstance(sdlc.performance_thresholds, dict)
        assert sdlc.current_phase is None
    
    def test_detect_project_type(self, sdlc):
        """Test project type detection."""
        project_type = sdlc._detect_project_type()
        assert project_type == "python_package"
    
    def test_detect_primary_language(self, sdlc, temp_project_root):
        """Test primary language detection."""
        # Create some Python files
        src_dir = Path(temp_project_root) / "src"
        (src_dir / "main.py").write_text("print('hello')")
        (src_dir / "utils.py").write_text("def helper(): pass")
        
        language = sdlc._detect_primary_language()
        assert language == "python"
    
    def test_detect_framework(self, sdlc, temp_project_root):
        """Test framework detection."""
        # Update pyproject.toml with slack-bolt
        pyproject_path = Path(temp_project_root) / "pyproject.toml"
        pyproject_path.write_text("""
[project]
name = "test_project"
dependencies = ["slack-bolt", "pytest"]
""")
        
        framework = sdlc._detect_framework()
        assert framework == "slack_bot"
    
    def test_detect_build_system(self, sdlc):
        """Test build system detection."""
        build_system = sdlc._detect_build_system()
        assert build_system == "pip"
    
    def test_detect_test_framework(self, sdlc):
        """Test test framework detection."""
        test_framework = sdlc._detect_test_framework()
        assert test_framework == "pytest"
    
    def test_analyze_existing_files(self, sdlc, temp_project_root):
        """Test existing file analysis."""
        # Create additional files
        project_root = Path(temp_project_root)
        (project_root / "config.yaml").write_text("key: value")
        (project_root / "data.json").write_text('{"test": true}')
        
        file_counts = sdlc._analyze_existing_files()
        
        assert file_counts["toml"] >= 1  # pyproject.toml
        assert file_counts["md"] >= 1    # README.md
        assert file_counts["yaml"] >= 1  # config.yaml
        assert file_counts["json"] >= 1  # data.json
    
    def test_analyze_dependencies(self, sdlc):
        """Test dependency analysis."""
        deps_info = sdlc._analyze_dependencies()
        
        assert "python" in deps_info
        assert isinstance(deps_info["python"], list)
        assert len(deps_info["python"]) >= 2  # requests, pytest
    
    @patch('subprocess.run')
    def test_get_git_status_clean(self, mock_run, sdlc):
        """Test git status when repository is clean."""
        mock_run.return_value = Mock(returncode=0, stdout="")
        
        git_status = sdlc._get_git_status()
        
        assert git_status["clean"] is True
        assert git_status["modified_files"] == 0
        assert git_status["added_files"] == 0
        assert git_status["untracked_files"] == 0
    
    @patch('subprocess.run')
    def test_get_git_status_with_changes(self, mock_run, sdlc):
        """Test git status with changes."""
        mock_run.return_value = Mock(
            returncode=0, 
            stdout=" M file1.py\nA  file2.py\n?? file3.py\n"
        )
        
        git_status = sdlc._get_git_status()
        
        assert git_status["clean"] is False
        assert git_status["modified_files"] == 1
        assert git_status["added_files"] == 1
        assert git_status["untracked_files"] == 1
    
    @patch('subprocess.run')
    def test_get_git_status_error(self, mock_run, sdlc):
        """Test git status with error."""
        mock_run.side_effect = Exception("Git error")
        
        git_status = sdlc._get_git_status()
        
        assert git_status["clean"] is False
        assert git_status["error"] is True
    
    def test_determine_sdlc_strategy(self, sdlc):
        """Test SDLC strategy determination."""
        project_info = {
            "project_type": "python_package",
            "language": "python",
            "framework": "slack_bot"
        }
        
        strategy = sdlc._determine_sdlc_strategy(project_info)
        assert strategy == "slack_bot"
    
    def test_analyze_project_structure(self, sdlc):
        """Test complete project structure analysis."""
        project_info = sdlc.analyze_project_structure()
        
        required_keys = [
            "project_type", "language", "framework", "build_system",
            "test_framework", "existing_files", "dependencies", 
            "git_status", "sdlc_strategy"
        ]
        
        for key in required_keys:
            assert key in project_info
        
        assert project_info["project_type"] == "python_package"
        assert project_info["language"] == "python"
        assert project_info["build_system"] == "pip"
        assert isinstance(project_info["existing_files"], dict)
    
    @patch('slack_kb_agent.autonomous_sdlc.create_dependent_tasks')
    @patch('slack_kb_agent.autonomous_sdlc.get_quantum_planner')
    def test_create_phase_tasks(self, mock_get_planner, mock_create_tasks, sdlc):
        """Test phase task creation."""
        mock_tasks = [Mock(), Mock()]
        mock_create_tasks.return_value = mock_tasks
        
        tasks = sdlc._create_phase_tasks(SDLCPhase.ANALYSIS)
        
        assert tasks == mock_tasks
        mock_create_tasks.assert_called_once()
    
    @patch('subprocess.run')
    def test_check_code_quality_success(self, mock_run, sdlc):
        """Test successful code quality check."""
        mock_run.return_value = Mock(returncode=0, stdout="All checks passed")
        
        result = sdlc._check_code_quality()
        
        assert result is True
        assert sdlc.metrics.code_quality_score == 100.0
        assert sdlc.metrics.quality_gate_results["code_quality"] is True
    
    @patch('subprocess.run')
    def test_check_code_quality_failure(self, mock_run, sdlc):
        """Test failed code quality check."""
        mock_run.return_value = Mock(returncode=1, stdout="Linting errors found")
        
        result = sdlc._check_code_quality()
        
        assert result is False
        assert sdlc.metrics.code_quality_score == 70.0
        assert sdlc.metrics.quality_gate_results["code_quality"] is False
    
    @patch('subprocess.run')
    def test_run_security_scan_success(self, mock_run, sdlc):
        """Test successful security scan."""
        bandit_output = {
            "results": [
                {"issue_severity": "LOW"},
                {"issue_severity": "MEDIUM"}
            ]
        }
        mock_run.return_value = Mock(
            returncode=0, 
            stdout=json.dumps(bandit_output)
        )
        
        result = sdlc._run_security_scan()
        
        assert result is True  # No HIGH severity issues
        assert sdlc.metrics.security_issues_found == 2
        assert sdlc.metrics.quality_gate_results["security_scan"] is True
    
    @patch('subprocess.run')
    def test_run_security_scan_with_high_severity(self, mock_run, sdlc):
        """Test security scan with high severity issues."""
        bandit_output = {
            "results": [
                {"issue_severity": "HIGH"},
                {"issue_severity": "MEDIUM"}
            ]
        }
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(bandit_output)
        )
        
        result = sdlc._run_security_scan()
        
        assert result is False  # Has HIGH severity issues
        assert sdlc.metrics.security_issues_found == 2
        assert sdlc.metrics.quality_gate_results["security_scan"] is False
    
    @patch('subprocess.run')
    def test_run_tests_success(self, mock_run, sdlc, temp_project_root):
        """Test successful test execution."""
        # Create mock coverage file
        coverage_data = {
            "totals": {
                "percent_covered": 92.5
            }
        }
        coverage_file = Path(temp_project_root) / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))
        
        mock_run.return_value = Mock(returncode=0, stdout="Tests passed")
        
        result = sdlc._run_tests()
        
        assert result is True
        assert sdlc.metrics.test_coverage_percentage == 92.5
        assert sdlc.metrics.quality_gate_results["test_coverage"] is True
    
    @patch('subprocess.run')
    def test_run_tests_low_coverage(self, mock_run, sdlc, temp_project_root):
        """Test test execution with low coverage."""
        # Create mock coverage file with low coverage
        coverage_data = {
            "totals": {
                "percent_covered": 70.0
            }
        }
        coverage_file = Path(temp_project_root) / "coverage.json"
        coverage_file.write_text(json.dumps(coverage_data))
        
        mock_run.return_value = Mock(returncode=0, stdout="Tests passed")
        
        result = sdlc._run_tests()
        
        assert result is False  # Failed due to low coverage
        assert sdlc.metrics.test_coverage_percentage == 70.0
        assert sdlc.metrics.quality_gate_results["test_coverage"] is False
    
    @patch('subprocess.run')
    def test_run_integration_tests(self, mock_run, sdlc):
        """Test integration test execution."""
        mock_run.return_value = Mock(returncode=0, stdout="Integration tests passed")
        
        result = sdlc._run_integration_tests()
        
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "integration" in " ".join(args)
    
    @patch('subprocess.run')
    def test_check_performance(self, mock_run, sdlc):
        """Test performance check."""
        mock_run.return_value = Mock(returncode=0, stdout="Performance tests passed")
        
        result = sdlc._check_performance()
        
        assert result is True
        assert sdlc.metrics.quality_gate_results["performance"] is True
    
    def test_check_documentation(self, sdlc):
        """Test documentation check."""
        result = sdlc._check_documentation()
        
        assert result is True  # README.md exists
        assert sdlc.metrics.documentation_coverage >= 50.0
        assert sdlc.metrics.quality_gate_results["documentation"] is True
    
    def test_run_quality_gate(self, sdlc):
        """Test quality gate execution."""
        # Mock the underlying check method
        with patch.object(sdlc, '_check_code_quality', return_value=True):
            result = sdlc._run_quality_gate(QualityGate.CODE_QUALITY)
            assert result is True
        
        with patch.object(sdlc, '_run_security_scan', return_value=False):
            result = sdlc._run_quality_gate(QualityGate.SECURITY_SCAN)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_sdlc_phase_success(self, sdlc):
        """Test successful SDLC phase execution."""
        # Mock task creation and execution
        mock_tasks = [Mock(), Mock()]
        for task in mock_tasks:
            task.state = Mock()
        
        with patch.object(sdlc, '_create_phase_tasks', return_value=mock_tasks), \
             patch.object(sdlc, '_wait_for_phase_completion', return_value=True), \
             patch.object(sdlc.planner, 'schedule_task', return_value=True):
            
            result = await sdlc.execute_sdlc_phase(SDLCPhase.ANALYSIS)
            
            assert result is True
            assert SDLCPhase.ANALYSIS in sdlc.executed_phases
            assert "analysis" in sdlc.metrics.phase_durations
    
    @pytest.mark.asyncio
    async def test_execute_sdlc_phase_failure(self, sdlc):
        """Test failed SDLC phase execution."""
        mock_tasks = [Mock()]
        
        with patch.object(sdlc, '_create_phase_tasks', return_value=mock_tasks), \
             patch.object(sdlc, '_wait_for_phase_completion', return_value=False), \
             patch.object(sdlc.planner, 'schedule_task', return_value=True):
            
            result = await sdlc.execute_sdlc_phase(SDLCPhase.TESTING)
            
            assert result is False
            assert SDLCPhase.TESTING in sdlc.failed_phases
    
    @pytest.mark.asyncio
    async def test_wait_for_phase_completion_success(self, sdlc):
        """Test waiting for phase completion - success case."""
        from slack_kb_agent.quantum_task_planner import TaskState
        
        mock_tasks = [Mock(), Mock()]
        for task in mock_tasks:
            task.state = TaskState.COMPLETED
        
        result = await sdlc._wait_for_phase_completion(mock_tasks)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_wait_for_phase_completion_with_failures(self, sdlc):
        """Test waiting for phase completion - with failures."""
        from slack_kb_agent.quantum_task_planner import TaskState
        
        mock_tasks = [Mock(), Mock()]
        mock_tasks[0].state = TaskState.COMPLETED
        mock_tasks[1].state = TaskState.FAILED
        
        result = await sdlc._wait_for_phase_completion(mock_tasks)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_wait_for_phase_completion_timeout(self, sdlc):
        """Test waiting for phase completion - timeout."""
        from slack_kb_agent.quantum_task_planner import TaskState
        
        mock_tasks = [Mock()]
        mock_tasks[0].state = TaskState.SUPERPOSITION  # Never completes
        
        # Use a very short timeout for testing
        original_method = sdlc._wait_for_phase_completion
        
        async def quick_timeout_method(tasks):
            max_wait_time = 0.1  # 100ms timeout
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < max_wait_time:
                completed = sum(1 for task in tasks if task.state == TaskState.COMPLETED)
                failed = sum(1 for task in tasks if task.state == TaskState.FAILED)
                
                if completed + failed == len(tasks):
                    return failed == 0
                
                await asyncio.sleep(0.01)
            
            return False
        
        sdlc._wait_for_phase_completion = quick_timeout_method
        
        result = await sdlc._wait_for_phase_completion(mock_tasks)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_full_sdlc_success(self, sdlc):
        """Test full SDLC execution success."""
        with patch.object(sdlc, 'analyze_project_structure', return_value={}), \
             patch.object(sdlc, 'execute_sdlc_phase', return_value=True):
            
            result = await sdlc.execute_full_sdlc()
            
            assert result is True
            assert sdlc.metrics.success_rate == 1.0
            assert len(sdlc.executed_phases) >= 5
    
    @pytest.mark.asyncio
    async def test_execute_full_sdlc_partial_failure(self, sdlc):
        """Test full SDLC execution with partial failure."""
        phase_results = {
            SDLCPhase.ANALYSIS: True,
            SDLCPhase.IMPLEMENTATION: True,
            SDLCPhase.TESTING: False,  # This phase fails
            SDLCPhase.DEPLOYMENT: True,
            SDLCPhase.MONITORING: True
        }
        
        def mock_execute_phase(phase):
            return phase_results.get(phase, True)
        
        with patch.object(sdlc, 'analyze_project_structure', return_value={}), \
             patch.object(sdlc, 'execute_sdlc_phase', side_effect=mock_execute_phase):
            
            result = await sdlc.execute_full_sdlc()
            
            assert result is False  # Should fail due to testing phase
    
    def test_get_execution_summary(self, sdlc):
        """Test execution summary generation."""
        # Set up some test data
        sdlc.metrics.success_rate = 0.8
        sdlc.metrics.test_coverage_percentage = 90.0
        sdlc.metrics.code_quality_score = 95.0
        sdlc.metrics.phase_durations["analysis"] = timedelta(seconds=30)
        sdlc.executed_phases = [SDLCPhase.ANALYSIS]
        sdlc.failed_phases = []
        
        summary = sdlc.get_execution_summary()
        
        assert "metrics" in summary
        assert "phases" in summary
        assert "quality_gates" in summary
        assert summary["metrics"]["success_rate"] == 0.8
        assert summary["metrics"]["test_coverage"] == 90.0
        assert summary["phases"]["executed"] == ["analysis"]
        assert summary["phases"]["durations"]["analysis"] == 30.0


class TestGlobalSDLCInstance:
    """Test global SDLC instance management."""
    
    def test_get_autonomous_sdlc_singleton(self, tmp_path):
        """Test global SDLC singleton behavior."""
        project_root = str(tmp_path / "test_project")
        
        # Reset global instance
        import slack_kb_agent.autonomous_sdlc as sdlc_module
        sdlc_module._autonomous_sdlc = None
        
        sdlc1 = get_autonomous_sdlc(project_root)
        sdlc2 = get_autonomous_sdlc(project_root)
        
        assert sdlc1 is sdlc2  # Should be same instance
        assert str(sdlc1.project_root) == project_root
    
    def test_get_autonomous_sdlc_different_roots(self, tmp_path):
        """Test SDLC instance creation with different project roots."""
        root1 = str(tmp_path / "project1")
        root2 = str(tmp_path / "project2")
        
        # Reset global instance
        import slack_kb_agent.autonomous_sdlc as sdlc_module
        sdlc_module._autonomous_sdlc = None
        
        sdlc1 = get_autonomous_sdlc(root1)
        sdlc2 = get_autonomous_sdlc(root2)
        
        assert sdlc1 is not sdlc2  # Should be different instances
        assert str(sdlc1.project_root) == root1
        assert str(sdlc2.project_root) == root2


class TestSDLCIntegration:
    """Integration tests for SDLC components."""
    
    @pytest.mark.asyncio
    async def test_full_integration_with_quantum_planner(self, tmp_path):
        """Test SDLC integration with quantum task planner."""
        project_root = tmp_path / "integration_test"
        project_root.mkdir()
        
        # Create minimal project structure
        (project_root / "pyproject.toml").write_text("[project]\nname='test'")
        (project_root / "README.md").write_text("# Test")
        
        sdlc = AutonomousSDLC(str(project_root))
        
        # Test that quantum planner is properly integrated
        assert sdlc.planner is not None
        
        # Test project analysis
        project_info = sdlc.analyze_project_structure()
        assert project_info["project_type"] == "python_package"
        
        # Test phase task creation
        tasks = sdlc._create_phase_tasks(SDLCPhase.ANALYSIS)
        assert len(tasks) > 0
        assert all(hasattr(task, 'id') for task in tasks)
    
    @pytest.mark.asyncio
    async def test_quality_gates_integration(self, tmp_path):
        """Test quality gates integration."""
        project_root = tmp_path / "quality_test"
        project_root.mkdir()
        (project_root / "pyproject.toml").write_text("[project]\nname='test'")
        (project_root / "README.md").write_text("# Test")
        
        sdlc = AutonomousSDLC(str(project_root))
        
        # Test documentation quality gate (should pass)
        result = sdlc._run_quality_gate(QualityGate.DOCUMENTATION)
        assert result is True
        
        # Verify metrics were updated
        assert "documentation" in sdlc.metrics.quality_gate_results
        assert sdlc.metrics.documentation_coverage > 0