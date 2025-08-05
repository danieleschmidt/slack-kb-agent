#!/usr/bin/env python3
"""
Quality Gate Runner

Autonomous quality gate execution for the enhanced Slack KB Agent.
Validates code quality, security, performance, and deployment readiness.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from slack_kb_agent.autonomous_sdlc import get_autonomous_sdlc, QualityGate
    from slack_kb_agent.performance_optimizer import get_performance_optimizer
    from slack_kb_agent.resilience import get_health_monitor
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Running basic quality checks instead...")

class QualityGateRunner:
    """Execute comprehensive quality gates for the project."""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.results = {}
        
    def run_import_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate that all new modules can be imported."""
        print("🔍 Validating module imports...")
        
        modules_to_test = [
            "slack_kb_agent.quantum_task_planner",
            "slack_kb_agent.autonomous_sdlc", 
            "slack_kb_agent.resilience",
            "slack_kb_agent.performance_optimizer"
        ]
        
        results = {}
        all_passed = True
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
                results[module_name] = {"status": "passed", "error": None}
                print(f"  ✅ {module_name}")
            except Exception as e:
                results[module_name] = {"status": "failed", "error": str(e)}
                print(f"  ❌ {module_name}: {e}")
                all_passed = False
        
        return all_passed, results
    
    def run_syntax_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate Python syntax for all new files."""
        print("🔍 Validating Python syntax...")
        
        python_files = [
            "src/slack_kb_agent/quantum_task_planner.py",
            "src/slack_kb_agent/autonomous_sdlc.py",
            "src/slack_kb_agent/resilience.py", 
            "src/slack_kb_agent/performance_optimizer.py"
        ]
        
        results = {}
        all_passed = True
        
        for file_path in python_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                results[file_path] = {"status": "skipped", "reason": "file_not_found"}
                continue
                
            try:
                with open(full_path, 'r') as f:
                    source_code = f.read()
                
                # Compile to check syntax
                compile(source_code, str(full_path), 'exec')
                results[file_path] = {"status": "passed", "lines": len(source_code.split('\n'))}
                print(f"  ✅ {file_path}")
                
            except SyntaxError as e:
                results[file_path] = {
                    "status": "failed", 
                    "error": f"Syntax error at line {e.lineno}: {e.msg}"
                }
                print(f"  ❌ {file_path}: Syntax Error")
                all_passed = False
            except Exception as e:
                results[file_path] = {"status": "failed", "error": str(e)}
                print(f"  ❌ {file_path}: {e}")
                all_passed = False
        
        return all_passed, results
    
    def run_basic_functionality_test(self) -> Tuple[bool, Dict[str, Any]]:
        """Test basic functionality of new components."""
        print("🔍 Testing basic functionality...")
        
        results = {}
        all_passed = True
        
        try:
            # Test quantum task planner
            print("  Testing quantum task planner...")
            from slack_kb_agent.quantum_task_planner import QuantumTask, TaskState, TaskPriority
            
            task = QuantumTask(name="test_task", priority=TaskPriority.HIGH)
            assert task.state == TaskState.SUPERPOSITION
            assert task.priority == TaskPriority.HIGH
            
            results["quantum_task_planner"] = {"status": "passed", "tests": ["task_creation", "state_enum", "priority_enum"]}
            print("    ✅ Quantum task planner basic tests passed")
            
        except Exception as e:
            results["quantum_task_planner"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Quantum task planner failed: {e}")
            all_passed = False
        
        try:
            # Test autonomous SDLC
            print("  Testing autonomous SDLC...")
            from slack_kb_agent.autonomous_sdlc import AutonomousSDLC, SDLCPhase, QualityGate
            
            sdlc = AutonomousSDLC(str(self.project_root))
            assert sdlc.project_root == self.project_root
            assert sdlc.min_test_coverage == 85.0
            
            # Test project analysis
            project_info = sdlc.analyze_project_structure()
            assert isinstance(project_info, dict)
            assert "project_type" in project_info
            
            results["autonomous_sdlc"] = {"status": "passed", "tests": ["initialization", "project_analysis"]}
            print("    ✅ Autonomous SDLC basic tests passed")
            
        except Exception as e:
            results["autonomous_sdlc"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Autonomous SDLC failed: {e}")
            all_passed = False
        
        try:
            # Test resilience components
            print("  Testing resilience components...")
            from slack_kb_agent.resilience import RetryConfig, BackoffStrategy, HealthStatus
            
            config = RetryConfig(max_attempts=3, backoff_strategy=BackoffStrategy.EXPONENTIAL)
            assert config.max_attempts == 3
            assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL
            
            # Test delay calculation
            delay1 = config.calculate_delay(1)
            delay2 = config.calculate_delay(2)
            assert delay2 > delay1  # Exponential should increase
            
            results["resilience"] = {"status": "passed", "tests": ["retry_config", "backoff_calculation"]}
            print("    ✅ Resilience components basic tests passed")
            
        except Exception as e:
            results["resilience"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Resilience components failed: {e}")
            all_passed = False
        
        try:
            # Test performance optimizer
            print("  Testing performance optimizer...")
            from slack_kb_agent.performance_optimizer import PerformanceMetrics, OptimizationStrategy
            
            metrics = PerformanceMetrics()
            assert metrics.cpu_usage == 0.0
            assert metrics.memory_usage == 0.0
            
            # Test conversion to dict
            metrics_dict = metrics.to_dict()
            assert isinstance(metrics_dict, dict)
            assert "timestamp" in metrics_dict
            
            results["performance_optimizer"] = {"status": "passed", "tests": ["metrics_creation", "dict_conversion"]}
            print("    ✅ Performance optimizer basic tests passed")
            
        except Exception as e:
            results["performance_optimizer"] = {"status": "failed", "error": str(e)}
            print(f"    ❌ Performance optimizer failed: {e}")
            all_passed = False
        
        return all_passed, results
    
    def run_documentation_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check documentation completeness."""
        print("🔍 Checking documentation...")
        
        required_docs = [
            "README.md",
            "ARCHITECTURE.md", 
            "API_USAGE_GUIDE.md"
        ]
        
        results = {}
        all_passed = True
        
        for doc in required_docs:
            doc_path = self.project_root / doc
            if doc_path.exists():
                size = doc_path.stat().st_size
                results[doc] = {"status": "passed", "size": size}
                print(f"  ✅ {doc} ({size} bytes)")
            else:
                results[doc] = {"status": "missing"}
                print(f"  ❌ {doc} - Missing")
                all_passed = False
        
        return all_passed, results
    
    def run_file_structure_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate project file structure."""
        print("🔍 Validating file structure...")
        
        expected_files = [
            "src/slack_kb_agent/__init__.py",
            "src/slack_kb_agent/quantum_task_planner.py",
            "src/slack_kb_agent/autonomous_sdlc.py",
            "src/slack_kb_agent/resilience.py",
            "src/slack_kb_agent/performance_optimizer.py",
            "tests/test_quantum_task_planner.py",
            "tests/test_autonomous_sdlc.py",
            "autonomous_execution_demo.py"
        ]
        
        results = {}
        all_passed = True
        total_lines = 0
        
        for file_path in expected_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        lines = len(f.readlines())
                    
                    results[file_path] = {"status": "exists", "lines": lines}
                    total_lines += lines
                    print(f"  ✅ {file_path} ({lines} lines)")
                except Exception as e:
                    results[file_path] = {"status": "error", "error": str(e)}
                    print(f"  ⚠️  {file_path} - Error reading: {e}")
            else:
                results[file_path] = {"status": "missing"}
                print(f"  ❌ {file_path} - Missing")
                all_passed = False
        
        results["summary"] = {"total_lines": total_lines, "files_checked": len(expected_files)}
        print(f"  📊 Total lines of new code: {total_lines}")
        
        return all_passed, results
    
    def run_integration_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate integration between components."""
        print("🔍 Validating component integration...")
        
        results = {}
        all_passed = True
        
        try:
            # Test that quantum planner integrates with SDLC
            from slack_kb_agent.autonomous_sdlc import get_autonomous_sdlc
            from slack_kb_agent.quantum_task_planner import get_quantum_planner
            
            sdlc = get_autonomous_sdlc(str(self.project_root))
            planner = get_quantum_planner()
            
            # Verify SDLC has access to planner
            assert sdlc.planner is not None
            assert hasattr(sdlc.planner, 'create_task')
            
            results["sdlc_planner_integration"] = {"status": "passed"}
            print("  ✅ SDLC-Planner integration")
            
        except Exception as e:
            results["sdlc_planner_integration"] = {"status": "failed", "error": str(e)}
            print(f"  ❌ SDLC-Planner integration failed: {e}")
            all_passed = False
        
        try:
            # Test that all components can be imported from main package
            from slack_kb_agent import (
                QuantumTaskPlanner, AutonomousSDLC, ResilientExecutor
            )
            
            results["package_imports"] = {"status": "passed"}
            print("  ✅ Package import integration")
            
        except Exception as e:
            results["package_imports"] = {"status": "failed", "error": str(e)}
            print(f"  ❌ Package import integration failed: {e}")
            all_passed = False
        
        return all_passed, results
    
    def run_demo_validation(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate the autonomous execution demo."""
        print("🔍 Validating demo script...")
        
        results = {}
        
        demo_path = self.project_root / "autonomous_execution_demo.py"
        if not demo_path.exists():
            results["demo_script"] = {"status": "missing"}
            print("  ❌ Demo script missing")
            return False, results
        
        try:
            # Check syntax
            with open(demo_path, 'r') as f:
                source_code = f.read()
            
            compile(source_code, str(demo_path), 'exec')
            
            # Check for required functions
            required_functions = ["demo_quantum_task_planning", "demo_autonomous_sdlc", "main"]
            missing_functions = []
            
            for func in required_functions:
                if f"def {func}" not in source_code and f"async def {func}" not in source_code:
                    missing_functions.append(func)
            
            if missing_functions:
                results["demo_script"] = {
                    "status": "incomplete", 
                    "missing_functions": missing_functions
                }
                print(f"  ⚠️  Demo script missing functions: {missing_functions}")
                return False, results
            else:
                results["demo_script"] = {"status": "passed", "lines": len(source_code.split('\n'))}
                print("  ✅ Demo script validation passed")
                return True, results
            
        except Exception as e:
            results["demo_script"] = {"status": "failed", "error": str(e)}
            print(f"  ❌ Demo script validation failed: {e}")
            return False, results
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        print("🚀 Running Autonomous SDLC Quality Gates")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all quality gates
        gates = [
            ("Import Validation", self.run_import_validation),
            ("Syntax Validation", self.run_syntax_validation),
            ("Basic Functionality", self.run_basic_functionality_test),
            ("Documentation Check", self.run_documentation_check),
            ("File Structure", self.run_file_structure_validation),
            ("Integration Validation", self.run_integration_validation),
            ("Demo Validation", self.run_demo_validation)
        ]
        
        results = {}
        passed_gates = 0
        total_gates = len(gates)
        
        for gate_name, gate_func in gates:
            try:
                passed, gate_results = gate_func()
                results[gate_name] = {
                    "passed": passed,
                    "results": gate_results
                }
                
                if passed:
                    passed_gates += 1
                    
            except Exception as e:
                results[gate_name] = {
                    "passed": False,
                    "error": str(e)
                }
                print(f"  💥 {gate_name} crashed: {e}")
        
        execution_time = time.time() - start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("🏁 Quality Gate Execution Summary")
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
        print(f"✅ Passed: {passed_gates}/{total_gates} quality gates")
        print(f"📊 Success rate: {(passed_gates/total_gates)*100:.1f}%")
        
        if passed_gates == total_gates:
            print("🎉 ALL QUALITY GATES PASSED! 🎉")
            print("🚀 Ready for autonomous execution!")
        else:
            failed_gates = total_gates - passed_gates
            print(f"⚠️  {failed_gates} quality gates failed")
            print("🔧 Review failed gates before deployment")
        
        # Overall results
        overall_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": execution_time,
            "total_gates": total_gates,
            "passed_gates": passed_gates,
            "success_rate": (passed_gates/total_gates)*100,
            "overall_status": "PASSED" if passed_gates == total_gates else "FAILED",
            "gate_results": results
        }
        
        return overall_results


def main():
    """Main execution function."""
    runner = QualityGateRunner()
    results = runner.run_all_quality_gates()
    
    # Save results to file
    results_file = Path("/root/repo/quality_gate_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Return appropriate exit code
    if results["overall_status"] == "PASSED":
        print("\n✨ Quality gates completed successfully!")
        return 0
    else:
        print("\n❌ Some quality gates failed. Check results for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())