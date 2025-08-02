#!/usr/bin/env python3
"""
Repository Maintenance Automation Script

Performs automated maintenance tasks for the Slack KB Agent repository including:
- Dependency updates
- Code quality checks
- Repository cleanup
- Health monitoring
"""

import json
import subprocess
import sys
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import tempfile
from typing import Dict, List, Any, Optional
import requests


class RepositoryMaintenance:
    """Automated repository maintenance tasks."""
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.maintenance_log = []
    
    def log_action(self, action: str, details: str = "", success: bool = True):
        """Log maintenance actions."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "details": details,
            "success": success,
            "dry_run": self.dry_run
        }
        self.maintenance_log.append(log_entry)
        
        status = "✅" if success else "❌"
        prefix = "[DRY RUN] " if self.dry_run else ""
        print(f"{prefix}{status} {action}: {details}")
    
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a command with optional dry-run simulation."""
        if self.dry_run:
            print(f"[DRY RUN] Would run: {' '.join(command)}")
            return subprocess.CompletedProcess(command, 0, "", "")
        
        return subprocess.run(
            command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=check
        )
    
    def cleanup_cache_files(self) -> None:
        """Clean up cache and temporary files."""
        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/.ruff_cache",
            "**/htmlcov",
            "**/.coverage*",
            "**/coverage.xml",
            "**/*.log",
            "**/tmp",
            "**/temp",
            "**/.DS_Store",
            "**/Thumbs.db"
        ]
        
        cleaned_files = 0
        for pattern in cache_patterns:
            for path in self.project_root.glob(pattern):
                if path.exists():
                    try:
                        if not self.dry_run:
                            if path.is_dir():
                                shutil.rmtree(path)
                            else:
                                path.unlink()
                        cleaned_files += 1
                    except Exception as e:
                        self.log_action(f"Failed to remove {path}", str(e), False)
        
        self.log_action("Cleanup cache files", f"Removed {cleaned_files} cache files/directories")
    
    def update_dependencies(self) -> Dict[str, Any]:
        """Check for and apply dependency updates."""
        updates_info = {"security": [], "regular": [], "errors": []}
        
        try:
            # Check for security vulnerabilities
            safety_result = self.run_command(["safety", "check", "--json"], check=False)
            if safety_result.returncode == 0 and safety_result.stdout:
                safety_data = json.loads(safety_result.stdout)
                if safety_data:
                    updates_info["security"] = safety_data
                    self.log_action("Security vulnerabilities found", f"{len(safety_data)} vulnerabilities")
            
            # Check for outdated packages
            pip_check_result = self.run_command(["pip", "list", "--outdated", "--format=json"], check=False)
            if pip_check_result.returncode == 0 and pip_check_result.stdout:
                outdated_packages = json.loads(pip_check_result.stdout)
                updates_info["regular"] = outdated_packages
                self.log_action("Outdated packages found", f"{len(outdated_packages)} packages")
        
        except Exception as e:
            updates_info["errors"].append(str(e))
            self.log_action("Dependency check failed", str(e), False)
        
        return updates_info
    
    def run_security_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scanning."""
        scan_results = {"bandit": None, "safety": None, "secrets": None, "errors": []}
        
        try:
            # Bandit security linting
            bandit_result = self.run_command([
                "bandit", "-r", "src/", "-f", "json"
            ], check=False)
            
            if bandit_result.returncode == 0 and bandit_result.stdout:
                scan_results["bandit"] = json.loads(bandit_result.stdout)
                issues = len(scan_results["bandit"].get("results", []))
                self.log_action("Bandit security scan", f"{issues} security issues found")
        
        except Exception as e:
            scan_results["errors"].append(f"Bandit scan failed: {e}")
            self.log_action("Bandit scan failed", str(e), False)
        
        try:
            # Secret detection
            secrets_result = self.run_command([
                "detect-secrets", "scan", "--all-files"
            ], check=False)
            
            if secrets_result.returncode == 0:
                self.log_action("Secret detection scan", "No secrets detected")
            else:
                scan_results["secrets"] = secrets_result.stdout
                self.log_action("Secret detection scan", "Potential secrets found", False)
        
        except Exception as e:
            scan_results["errors"].append(f"Secret detection failed: {e}")
            self.log_action("Secret detection failed", str(e), False)
        
        return scan_results
    
    def code_quality_check(self) -> Dict[str, Any]:
        """Run code quality checks and auto-fix when possible."""
        quality_results = {"formatting": False, "linting": False, "type_checking": False, "errors": []}
        
        try:
            # Code formatting with Black
            black_result = self.run_command(["black", "--check", "src/", "tests/"], check=False)
            if black_result.returncode != 0:
                # Auto-fix formatting issues
                fix_result = self.run_command(["black", "src/", "tests/"], check=False)
                quality_results["formatting"] = fix_result.returncode == 0
                self.log_action("Code formatting", "Auto-fixed formatting issues")
            else:
                quality_results["formatting"] = True
                self.log_action("Code formatting", "No formatting issues")
        
        except Exception as e:
            quality_results["errors"].append(f"Black formatting failed: {e}")
            self.log_action("Code formatting failed", str(e), False)
        
        try:
            # Linting with Ruff
            ruff_result = self.run_command(["ruff", "check", "--fix", "src/", "tests/"], check=False)
            quality_results["linting"] = ruff_result.returncode == 0
            
            if ruff_result.returncode == 0:
                self.log_action("Code linting", "No linting issues")
            else:
                self.log_action("Code linting", "Linting issues found and fixed where possible")
        
        except Exception as e:
            quality_results["errors"].append(f"Ruff linting failed: {e}")
            self.log_action("Code linting failed", str(e), False)
        
        try:
            # Type checking with MyPy
            mypy_result = self.run_command(["mypy", "src/slack_kb_agent/"], check=False)
            quality_results["type_checking"] = mypy_result.returncode == 0
            
            if mypy_result.returncode == 0:
                self.log_action("Type checking", "No type errors")
            else:
                error_count = mypy_result.stdout.count("error:")
                self.log_action("Type checking", f"{error_count} type errors found", False)
        
        except Exception as e:
            quality_results["errors"].append(f"MyPy type checking failed: {e}")
            self.log_action("Type checking failed", str(e), False)
        
        return quality_results
    
    def run_tests(self) -> Dict[str, Any]:
        """Run test suite and collect results."""
        test_results = {"passed": False, "coverage": 0, "duration": 0, "errors": []}
        
        try:
            start_time = datetime.utcnow()
            
            # Run tests with coverage
            test_result = self.run_command([
                "python", "-m", "pytest", 
                "tests/", 
                "--cov=src/slack_kb_agent",
                "--cov-report=json",
                "--cov-report=term",
                "-v"
            ], check=False)
            
            end_time = datetime.utcnow()
            test_results["duration"] = (end_time - start_time).total_seconds()
            test_results["passed"] = test_result.returncode == 0
            
            # Extract coverage information
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    test_results["coverage"] = coverage_data["totals"]["percent_covered"]
            
            if test_results["passed"]:
                self.log_action("Test suite", f"All tests passed, {test_results['coverage']:.1f}% coverage")
            else:
                self.log_action("Test suite", "Some tests failed", False)
        
        except Exception as e:
            test_results["errors"].append(str(e))
            self.log_action("Test execution failed", str(e), False)
        
        return test_results
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness and quality."""
        doc_results = {"readme_updated": False, "changelog_updated": False, "api_docs": False, "errors": []}
        
        try:
            # Check if README.md is up to date
            readme_file = self.project_root / "README.md"
            if readme_file.exists():
                readme_stat = readme_file.stat()
                week_ago = datetime.utcnow().timestamp() - (7 * 24 * 3600)
                doc_results["readme_updated"] = readme_stat.st_mtime > week_ago
            
            # Check if CHANGELOG.md exists and is updated
            changelog_file = self.project_root / "CHANGELOG.md"
            if changelog_file.exists():
                changelog_stat = changelog_file.stat()
                month_ago = datetime.utcnow().timestamp() - (30 * 24 * 3600)
                doc_results["changelog_updated"] = changelog_stat.st_mtime > month_ago
            
            # Check for API documentation
            api_docs_patterns = ["docs/api/", "docs/*.md", "*.md"]
            doc_count = 0
            for pattern in api_docs_patterns:
                doc_count += len(list(self.project_root.glob(pattern)))
            
            doc_results["api_docs"] = doc_count > 5  # Threshold for adequate documentation
            
            self.log_action("Documentation check", f"Found {doc_count} documentation files")
        
        except Exception as e:
            doc_results["errors"].append(str(e))
            self.log_action("Documentation check failed", str(e), False)
        
        return doc_results
    
    def cleanup_git_branches(self) -> Dict[str, Any]:
        """Clean up merged and stale Git branches."""
        cleanup_results = {"merged_branches": [], "stale_branches": [], "errors": []}
        
        try:
            # Get list of merged branches
            merged_result = self.run_command(["git", "branch", "--merged", "main"], check=False)
            if merged_result.returncode == 0:
                merged_branches = [
                    branch.strip() for branch in merged_result.stdout.split('\n')
                    if branch.strip() and not branch.strip().startswith('*') and 'main' not in branch
                ]
                
                # Delete merged branches (except main/develop)
                for branch in merged_branches:
                    if branch not in ['main', 'develop', 'master']:
                        delete_result = self.run_command(["git", "branch", "-d", branch], check=False)
                        if delete_result.returncode == 0:
                            cleanup_results["merged_branches"].append(branch)
                
                self.log_action("Git branch cleanup", f"Removed {len(cleanup_results['merged_branches'])} merged branches")
            
            # Identify stale branches (older than 30 days)
            branch_list_result = self.run_command(["git", "for-each-ref", "--format=%(refname:short) %(committerdate)", "refs/heads/"], check=False)
            if branch_list_result.returncode == 0:
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                
                for line in branch_list_result.stdout.split('\n'):
                    if line.strip():
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            branch_name, commit_date = parts
                            try:
                                commit_datetime = datetime.strptime(commit_date, '%Y-%m-%d %H:%M:%S %z')
                                if commit_datetime.replace(tzinfo=None) < thirty_days_ago:
                                    if branch_name not in ['main', 'develop', 'master']:
                                        cleanup_results["stale_branches"].append(branch_name)
                            except ValueError:
                                pass  # Skip if date parsing fails
                
                self.log_action("Stale branch detection", f"Found {len(cleanup_results['stale_branches'])} stale branches")
        
        except Exception as e:
            cleanup_results["errors"].append(str(e))
            self.log_action("Git cleanup failed", str(e), False)
        
        return cleanup_results
    
    def generate_maintenance_report(self) -> Dict[str, Any]:
        """Generate comprehensive maintenance report."""
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "project": "slack-kb-agent",
            "maintenance_run": {
                "dry_run": self.dry_run,
                "duration": 0,
                "total_actions": len(self.maintenance_log),
                "successful_actions": len([log for log in self.maintenance_log if log["success"]]),
                "failed_actions": len([log for log in self.maintenance_log if not log["success"]])
            },
            "actions_log": self.maintenance_log,
            "recommendations": []
        }
        
        # Generate recommendations based on results
        failed_actions = [log for log in self.maintenance_log if not log["success"]]
        if failed_actions:
            report["recommendations"].append({
                "priority": "high",
                "action": "investigate_failures",
                "description": f"Investigate {len(failed_actions)} failed maintenance actions"
            })
        
        # Check for security issues
        security_logs = [log for log in self.maintenance_log if "security" in log["action"].lower()]
        if any("found" in log["details"] for log in security_logs):
            report["recommendations"].append({
                "priority": "critical",
                "action": "address_security_issues",
                "description": "Address security vulnerabilities found during scan"
            })
        
        return report
    
    def run_full_maintenance(self) -> Dict[str, Any]:
        """Run complete maintenance routine."""
        start_time = datetime.utcnow()
        
        print(f"Starting repository maintenance {'(DRY RUN)' if self.dry_run else ''}...")
        
        # Run maintenance tasks
        self.cleanup_cache_files()
        dependency_updates = self.update_dependencies()
        security_scan = self.run_security_scan()
        quality_check = self.code_quality_check()
        test_results = self.run_tests()
        doc_check = self.check_documentation()
        git_cleanup = self.cleanup_git_branches()
        
        # Generate report
        report = self.generate_maintenance_report()
        report["maintenance_run"]["duration"] = (datetime.utcnow() - start_time).total_seconds()
        
        # Include detailed results
        report["results"] = {
            "dependency_updates": dependency_updates,
            "security_scan": security_scan,
            "quality_check": quality_check,
            "test_results": test_results,
            "documentation_check": doc_check,
            "git_cleanup": git_cleanup
        }
        
        print(f"Maintenance completed in {report['maintenance_run']['duration']:.1f} seconds")
        print(f"Total actions: {report['maintenance_run']['total_actions']}")
        print(f"Successful: {report['maintenance_run']['successful_actions']}")
        print(f"Failed: {report['maintenance_run']['failed_actions']}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Automated repository maintenance")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                      help="Path to project root directory")
    parser.add_argument("--dry-run", action="store_true",
                      help="Run maintenance checks without making changes")
    parser.add_argument("--output-report", type=str,
                      help="Output file for maintenance report")
    parser.add_argument("--tasks", nargs="+", 
                      choices=["cleanup", "dependencies", "security", "quality", "tests", "docs", "git"],
                      help="Specific maintenance tasks to run")
    
    args = parser.parse_args()
    
    maintenance = RepositoryMaintenance(args.project_root, args.dry_run)
    
    try:
        if args.tasks:
            # Run specific tasks
            report = {"actions_log": []}
            
            if "cleanup" in args.tasks:
                maintenance.cleanup_cache_files()
            if "dependencies" in args.tasks:
                maintenance.update_dependencies()
            if "security" in args.tasks:
                maintenance.run_security_scan()
            if "quality" in args.tasks:
                maintenance.code_quality_check()
            if "tests" in args.tasks:
                maintenance.run_tests()
            if "docs" in args.tasks:
                maintenance.check_documentation()
            if "git" in args.tasks:
                maintenance.cleanup_git_branches()
            
            report = maintenance.generate_maintenance_report()
        else:
            # Run full maintenance
            report = maintenance.run_full_maintenance()
        
        if args.output_report:
            with open(args.output_report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Maintenance report written to {args.output_report}")
        
        # Return appropriate exit code
        if report["maintenance_run"]["failed_actions"] > 0:
            return 1
        return 0
    
    except Exception as e:
        print(f"Error during maintenance: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())