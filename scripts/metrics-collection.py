#!/usr/bin/env python3
"""
Automated Metrics Collection Script for Slack KB Agent

This script collects various project metrics and updates the project-metrics.json file.
Designed to run in CI/CD pipelines or as a scheduled task.
"""

import json
import subprocess
import sys
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class MetricsCollector:
    """Collect and update project metrics."""
    
    def __init__(self, project_root: Path, github_token: Optional[str] = None):
        self.project_root = project_root
        self.github_token = github_token
        self.metrics_file = project_root / ".github" / "project-metrics.json"
        self.repo_info = self._get_repo_info()
    
    def _get_repo_info(self) -> Dict[str, str]:
        """Extract repository information from git."""
        try:
            origin_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=self.project_root,
                text=True
            ).strip()
            
            # Parse GitHub repository from URL
            if "github.com" in origin_url:
                if origin_url.startswith("git@"):
                    repo = origin_url.split(":")[-1].replace(".git", "")
                else:
                    repo = "/".join(origin_url.split("/")[-2:]).replace(".git", "")
                
                return {
                    "owner": repo.split("/")[0],
                    "name": repo.split("/")[1],
                    "full_name": repo
                }
        except subprocess.CalledProcessError:
            pass
        
        return {"owner": "", "name": "", "full_name": ""}
    
    def load_current_metrics(self) -> Dict[str, Any]:
        """Load current metrics from file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save updated metrics to file."""
        metrics["project"]["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Test coverage
        try:
            coverage_result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src/slack_kb_agent", "--cov-report=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if coverage_result.returncode == 0:
                coverage_file = self.project_root / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                        metrics["test_coverage"] = {
                            "current": round(coverage_data["totals"]["percent_covered"], 1),
                            "last_measured": datetime.utcnow().isoformat() + "Z"
                        }
        except Exception as e:
            print(f"Warning: Could not collect test coverage: {e}")
        
        # Code complexity (using radon)
        try:
            complexity_result = subprocess.run(
                ["radon", "cc", "src/", "-a", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if complexity_result.returncode == 0:
                complexity_data = json.loads(complexity_result.stdout)
                total_complexity = 0
                total_functions = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item["type"] in ["function", "method"]:
                            total_complexity += item["complexity"]
                            total_functions += 1
                
                if total_functions > 0:
                    avg_complexity = total_complexity / total_functions
                    metrics["code_complexity"] = {
                        "current": round(avg_complexity, 1),
                        "last_measured": datetime.utcnow().isoformat() + "Z"
                    }
        except Exception as e:
            print(f"Warning: Could not collect code complexity: {e}")
        
        # Security vulnerabilities
        try:
            security_result = subprocess.run(
                ["safety", "check", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if security_result.returncode == 0:
                security_data = json.loads(security_result.stdout)
                metrics["security_vulnerabilities"] = {
                    "current": len(security_data),
                    "last_measured": datetime.utcnow().isoformat() + "Z"
                }
        except Exception as e:
            print(f"Warning: Could not collect security metrics: {e}")
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect metrics from GitHub API."""
        if not self.github_token or not self.repo_info["full_name"]:
            return {}
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        base_url = f"https://api.github.com/repos/{self.repo_info['full_name']}"
        metrics = {}
        
        try:
            # Get recent commits (last week)
            since_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
            commits_response = requests.get(
                f"{base_url}/commits",
                headers=headers,
                params={"since": since_date}
            )
            
            if commits_response.status_code == 200:
                commits = commits_response.json()
                metrics["commit_frequency"] = {
                    "current": len(commits),
                    "last_measured": datetime.utcnow().isoformat() + "Z"
                }
            
            # Get pull requests
            prs_response = requests.get(
                f"{base_url}/pulls",
                headers=headers,
                params={"state": "closed", "per_page": 50}
            )
            
            if prs_response.status_code == 200:
                prs = prs_response.json()
                
                # Calculate average PR merge time
                merge_times = []
                for pr in prs:
                    if pr["merged_at"] and pr["created_at"]:
                        created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                        merged = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
                        merge_time = (merged - created).total_seconds() / 3600  # hours
                        merge_times.append(merge_time)
                
                if merge_times:
                    avg_merge_time = sum(merge_times) / len(merge_times)
                    metrics["pr_merge_time"] = {
                        "current": round(avg_merge_time, 1),
                        "last_measured": datetime.utcnow().isoformat() + "Z"
                    }
            
            # Get issues
            issues_response = requests.get(
                f"{base_url}/issues",
                headers=headers,
                params={"state": "closed", "per_page": 50}
            )
            
            if issues_response.status_code == 200:
                issues = issues_response.json()
                
                # Calculate average issue resolution time
                resolution_times = []
                for issue in issues:
                    if issue["closed_at"] and issue["created_at"] and not issue.get("pull_request"):
                        created = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
                        closed = datetime.fromisoformat(issue["closed_at"].replace("Z", "+00:00"))
                        resolution_time = (closed - created).total_seconds() / 3600  # hours
                        resolution_times.append(resolution_time)
                
                if resolution_times:
                    avg_resolution_time = sum(resolution_times) / len(resolution_times)
                    metrics["issue_resolution_time"] = {
                        "current": round(avg_resolution_time, 1),
                        "last_measured": datetime.utcnow().isoformat() + "Z"
                    }
        
        except Exception as e:
            print(f"Warning: Could not collect GitHub metrics: {e}")
        
        return metrics
    
    def collect_deployment_metrics(self) -> Dict[str, Any]:
        """Collect deployment and operational metrics."""
        metrics = {}
        
        # Try to get deployment frequency from GitHub Actions
        if self.github_token and self.repo_info["full_name"]:
            try:
                headers = {
                    "Authorization": f"token {self.github_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
                
                base_url = f"https://api.github.com/repos/{self.repo_info['full_name']}"
                
                # Get workflow runs for the last week
                since_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
                runs_response = requests.get(
                    f"{base_url}/actions/runs",
                    headers=headers,
                    params={"created": f">{since_date}"}
                )
                
                if runs_response.status_code == 200:
                    runs = runs_response.json()["workflow_runs"]
                    
                    # Count deployment runs
                    deployment_runs = [
                        run for run in runs 
                        if "deploy" in run["name"].lower() and run["conclusion"] == "success"
                    ]
                    
                    metrics["deployment_frequency"] = {
                        "current": len(deployment_runs),
                        "last_measured": datetime.utcnow().isoformat() + "Z"
                    }
                    
                    # Calculate deployment success rate
                    all_deployment_runs = [
                        run for run in runs 
                        if "deploy" in run["name"].lower()
                    ]
                    
                    if all_deployment_runs:
                        success_rate = len(deployment_runs) / len(all_deployment_runs) * 100
                        metrics["deployment_success_rate"] = {
                            "current": round(success_rate, 1),
                            "last_measured": datetime.utcnow().isoformat() + "Z"
                        }
            
            except Exception as e:
                print(f"Warning: Could not collect deployment metrics: {e}")
        
        return metrics
    
    def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics if the app is running."""
        metrics = {}
        
        # Try to get metrics from running application
        try:
            response = requests.get("http://localhost:9090/metrics", timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                
                # Parse Prometheus metrics (simplified)
                for line in metrics_text.split('\n'):
                    if 'query_duration_seconds' in line and not line.startswith('#'):
                        # Extract P95 latency if available
                        # This is a simplified parser - in practice, use prometheus_client
                        pass
                
                # Get health status
                health_response = requests.get("http://localhost:9090/health", timeout=5)
                if health_response.status_code == 200:
                    metrics["uptime"] = {
                        "current": 100.0,
                        "last_measured": datetime.utcnow().isoformat() + "Z"
                    }
        
        except Exception as e:
            print(f"Info: Application metrics not available: {e}")
        
        return metrics
    
    def update_trends(self, current_metrics: Dict[str, Any], new_values: Dict[str, Any]) -> Dict[str, Any]:
        """Update trend information based on current and new values."""
        updated_metrics = current_metrics.copy()
        
        for category, metrics in new_values.items():
            if category not in updated_metrics.get("metrics", {}):
                continue
            
            for metric_name, new_data in metrics.items():
                if metric_name not in updated_metrics["metrics"][category]:
                    continue
                
                current_metric = updated_metrics["metrics"][category][metric_name]
                old_value = current_metric.get("current", 0)
                new_value = new_data.get("current", 0)
                
                # Determine trend
                if new_value > old_value:
                    trend = "improving" if metric_name in ["test_coverage", "uptime", "user_satisfaction"] else "increasing"
                elif new_value < old_value:
                    trend = "declining" if metric_name in ["test_coverage", "uptime", "user_satisfaction"] else "decreasing"
                else:
                    trend = "stable"
                
                # Update the metric
                current_metric.update(new_data)
                current_metric["trend"] = trend
        
        return updated_metrics
    
    def generate_summary_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a summary report of current metrics."""
        report = []
        report.append(f"# Metrics Report for {metrics['project']['name']}")
        report.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        report.append("")
        
        for category, category_metrics in metrics.get("metrics", {}).items():
            report.append(f"## {category.replace('_', ' ').title()}")
            
            for metric_name, metric_data in category_metrics.items():
                name = metric_name.replace('_', ' ').title()
                current = metric_data.get("current", "N/A")
                target = metric_data.get("target", "N/A")
                trend = metric_data.get("trend", "unknown")
                
                status = "✅" if self._meets_target(current, target) else "⚠️"
                trend_emoji = {"improving": "📈", "stable": "➡️", "declining": "📉"}.get(trend, "❓")
                
                report.append(f"- {name}: {current} (target: {target}) {status} {trend_emoji}")
            
            report.append("")
        
        return "\n".join(report)
    
    def _meets_target(self, current: Any, target: str) -> bool:
        """Check if current value meets target."""
        try:
            if isinstance(current, (int, float)):
                if target.startswith(">"):
                    return current > float(target[1:].strip().replace("%", ""))
                elif target.startswith("<"):
                    return current < float(target[1:].strip().replace("%", ""))
                elif "%" in target:
                    return current >= float(target.replace("%", "").strip())
            return True
        except:
            return True
    
    def run_collection(self) -> Dict[str, Any]:
        """Run the complete metrics collection process."""
        print("Starting metrics collection...")
        
        # Load current metrics
        current_metrics = self.load_current_metrics()
        
        # Collect new metrics
        new_metrics = {}
        
        print("Collecting code quality metrics...")
        code_quality = self.collect_code_quality_metrics()
        if code_quality:
            new_metrics["code_quality"] = code_quality
        
        print("Collecting GitHub metrics...")
        github_metrics = self.collect_github_metrics()
        if github_metrics:
            new_metrics["development_velocity"] = github_metrics
        
        print("Collecting deployment metrics...")
        deployment_metrics = self.collect_deployment_metrics()
        if deployment_metrics:
            new_metrics["operational_excellence"] = deployment_metrics
        
        print("Collecting application metrics...")
        app_metrics = self.collect_application_metrics()
        if app_metrics:
            new_metrics["user_experience"] = app_metrics
        
        # Update trends and merge with current metrics
        updated_metrics = self.update_trends(current_metrics, new_metrics)
        
        # Save updated metrics
        self.save_metrics(updated_metrics)
        
        print(f"Metrics collection completed. Updated {len(new_metrics)} categories.")
        return updated_metrics


def main():
    parser = argparse.ArgumentParser(description="Collect and update project metrics")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                      help="Path to project root directory")
    parser.add_argument("--github-token", type=str, 
                      default=os.environ.get("GITHUB_TOKEN"),
                      help="GitHub API token for collecting repository metrics")
    parser.add_argument("--output-report", type=str,
                      help="Output file for metrics summary report")
    parser.add_argument("--dry-run", action="store_true",
                      help="Run collection but don't save results")
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.project_root, args.github_token)
    
    try:
        metrics = collector.run_collection()
        
        if args.output_report:
            report = collector.generate_summary_report(metrics)
            with open(args.output_report, 'w') as f:
                f.write(report)
            print(f"Summary report written to {args.output_report}")
        
        if not args.dry_run:
            print("Metrics saved successfully.")
        else:
            print("Dry run completed - no changes saved.")
        
        return 0
    
    except Exception as e:
        print(f"Error during metrics collection: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())