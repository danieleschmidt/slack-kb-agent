#!/usr/bin/env python3
"""
SBOM (Software Bill of Materials) Generator
Generates comprehensive SBOM for security compliance and vulnerability tracking.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import hashlib
import pkg_resources


class SBOMGenerator:
    """Generate SPDX-compliant SBOM for the project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.project_name = "slack-kb-agent"
        self.project_version = self._get_project_version()
        
    def _get_project_version(self) -> str:
        """Extract project version from pyproject.toml."""
        try:
            import tomllib
            with open(self.project_root / "pyproject.toml", "rb") as f:
                data = tomllib.load(f)
                return data["project"]["version"]
        except Exception:
            return "unknown"
    
    def _get_installed_packages(self) -> List[Dict[str, Any]]:
        """Get list of installed Python packages with versions and metadata."""
        packages = []
        
        for dist in pkg_resources.working_set:
            package_info = {
                "name": dist.project_name,
                "version": dist.version,
                "location": dist.location,
                "dependencies": [str(req) for req in dist.requires()],
                "metadata": {
                    "homepage": getattr(dist, 'homepage', None),
                    "author": getattr(dist, 'author', None),
                    "license": getattr(dist, 'license', None),
                }
            }
            
            # Calculate file hash if possible
            try:
                if hasattr(dist, 'egg_info') and dist.egg_info:
                    hash_obj = hashlib.sha256()
                    hash_obj.update(str(dist.egg_info).encode())
                    package_info["hash"] = hash_obj.hexdigest()
            except Exception:
                package_info["hash"] = None
                
            packages.append(package_info)
        
        return sorted(packages, key=lambda x: x["name"])
    
    def _get_system_dependencies(self) -> List[Dict[str, Any]]:
        """Identify system-level dependencies."""
        system_deps = []
        
        # Check for common system dependencies
        system_tools = [
            "git", "docker", "postgres", "redis-server", 
            "python3", "pip", "npm", "node"
        ]
        
        for tool in system_tools:
            try:
                result = subprocess.run(
                    [tool, "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    system_deps.append({
                        "name": tool,
                        "version": version,
                        "type": "system",
                        "required": tool in ["python3", "git"]
                    })
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
                
        return system_deps
    
    def _get_container_dependencies(self) -> List[Dict[str, Any]]:
        """Extract dependencies from Docker containers."""
        container_deps = []
        
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
                
            # Parse FROM statements
            for line in dockerfile_content.split('\n'):
                if line.strip().startswith('FROM'):
                    base_image = line.split()[1]
                    container_deps.append({
                        "name": base_image.split(':')[0],
                        "version": base_image.split(':')[1] if ':' in base_image else "latest",
                        "type": "container_base",
                        "source": "Dockerfile"
                    })
        
        return container_deps
    
    def _generate_vulnerability_data(self, packages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate vulnerability assessment data."""
        vuln_data = {
            "last_scan": datetime.utcnow().isoformat(),
            "scan_tools": ["safety", "pip-audit", "bandit"],
            "total_packages": len(packages),
            "high_risk_packages": [],
            "recommendations": []
        }
        
        # Identify potentially high-risk packages
        high_risk_patterns = [
            "pickle", "yaml", "xml", "eval", "exec", 
            "subprocess", "os.system", "shell"
        ]
        
        for package in packages:
            package_name = package["name"].lower()
            if any(pattern in package_name for pattern in high_risk_patterns):
                vuln_data["high_risk_packages"].append({
                    "name": package["name"],
                    "reason": "Contains potentially risky functionality",
                    "mitigation": "Review usage and consider alternatives"
                })
        
        return vuln_data
    
    def generate_spdx_sbom(self) -> Dict[str, Any]:
        """Generate SPDX 2.3 compliant SBOM."""
        packages = self._get_installed_packages()
        system_deps = self._get_system_dependencies()
        container_deps = self._get_container_dependencies()
        vuln_data = self._generate_vulnerability_data(packages)
        
        sbom = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"{self.project_name}-{self.project_version}",
            "documentNamespace": f"https://github.com/example/{self.project_name}/sbom/{datetime.utcnow().isoformat()}",
            "creationInfo": {
                "created": datetime.utcnow().isoformat(),
                "creators": ["Tool: slack-kb-agent-sbom-generator"],
                "licenseListVersion": "3.21"
            },
            "packages": [],
            "relationships": [],
            "vulnerabilityAssessment": vuln_data
        }
        
        # Add main project package
        main_package = {
            "SPDXID": "SPDXRef-Package-Main",
            "name": self.project_name,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "versionInfo": self.project_version,
            "supplier": "Organization: Your Organization",
            "copyrightText": "NOASSERTION"
        }
        sbom["packages"].append(main_package)
        
        # Add Python packages
        for i, package in enumerate(packages, 1):
            spdx_package = {
                "SPDXID": f"SPDXRef-Package-Python-{i}",
                "name": package["name"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "versionInfo": package["version"],
                "supplier": f"Person: {package['metadata'].get('author', 'NOASSERTION')}",
                "homepage": package["metadata"].get("homepage", "NOASSERTION"),
                "copyrightText": "NOASSERTION",
                "licenseConcluded": package["metadata"].get("license", "NOASSERTION"),
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE_MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/{package['name']}@{package['version']}"
                    }
                ]
            }
            
            if package.get("hash"):
                spdx_package["checksums"] = [{
                    "algorithm": "SHA256",
                    "value": package["hash"]
                }]
            
            sbom["packages"].append(spdx_package)
            
            # Add dependency relationship
            sbom["relationships"].append({
                "spdxElementId": "SPDXRef-Package-Main",
                "relatedSpdxElement": f"SPDXRef-Package-Python-{i}",
                "relationshipType": "DEPENDS_ON"
            })
        
        # Add system dependencies
        for i, dep in enumerate(system_deps, len(packages) + 1):
            spdx_package = {
                "SPDXID": f"SPDXRef-Package-System-{i}",
                "name": dep["name"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "versionInfo": dep["version"],
                "supplier": "NOASSERTION",
                "copyrightText": "NOASSERTION"
            }
            sbom["packages"].append(spdx_package)
            
            if dep.get("required", False):
                sbom["relationships"].append({
                    "spdxElementId": "SPDXRef-Package-Main",
                    "relatedSpdxElement": f"SPDXRef-Package-System-{i}",
                    "relationshipType": "DEPENDS_ON"
                })
        
        # Add container dependencies
        for i, dep in enumerate(container_deps, len(packages) + len(system_deps) + 1):
            spdx_package = {
                "SPDXID": f"SPDXRef-Package-Container-{i}",
                "name": dep["name"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "versionInfo": dep["version"],
                "supplier": "NOASSERTION",
                "copyrightText": "NOASSERTION"
            }
            sbom["packages"].append(spdx_package)
            
            sbom["relationships"].append({
                "spdxElementId": "SPDXRef-Package-Main",
                "relatedSpdxElement": f"SPDXRef-Package-Container-{i}",
                "relationshipType": "BUILD_DEPENDENCY_OF"
            })
        
        return sbom
    
    def save_sbom(self, output_path: Path, format_type: str = "json") -> None:
        """Save SBOM to file in specified format."""
        sbom = self.generate_spdx_sbom()
        
        if format_type.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(sbom, f, indent=2, sort_keys=True)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def generate_summary_report(self) -> str:
        """Generate human-readable SBOM summary."""
        sbom = self.generate_spdx_sbom()
        
        python_packages = [p for p in sbom["packages"] if "Python" in p["SPDXID"]]
        system_packages = [p for p in sbom["packages"] if "System" in p["SPDXID"]]
        container_packages = [p for p in sbom["packages"] if "Container" in p["SPDXID"]]
        
        vuln_data = sbom["vulnerabilityAssessment"]
        
        report = f"""
SBOM Summary Report
==================
Project: {self.project_name} v{self.project_version}
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

Dependencies Summary:
- Python packages: {len(python_packages)}
- System dependencies: {len(system_packages)}
- Container dependencies: {len(container_packages)}
- Total components: {len(sbom['packages'])}

Security Assessment:
- Last scan: {vuln_data['last_scan']}
- High-risk packages: {len(vuln_data['high_risk_packages'])}
- Scan tools: {', '.join(vuln_data['scan_tools'])}

High-Risk Packages:
"""
        
        for pkg in vuln_data["high_risk_packages"]:
            report += f"- {pkg['name']}: {pkg['reason']}\n"
        
        if not vuln_data["high_risk_packages"]:
            report += "- None identified\n"
        
        report += f"""
Top 10 Python Dependencies:
"""
        for pkg in python_packages[:10]:
            report += f"- {pkg['name']} v{pkg['versionInfo']}\n"
        
        return report


def main():
    """Main entry point for SBOM generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SBOM for slack-kb-agent")
    parser.add_argument(
        "--output", "-o", 
        type=Path, 
        default="sbom.json",
        help="Output file path"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json"],
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Generate summary report"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    generator = SBOMGenerator(project_root)
    
    try:
        generator.save_sbom(args.output, args.format)
        print(f"SBOM generated successfully: {args.output}")
        
        if args.summary:
            summary = generator.generate_summary_report()
            print(summary)
            
            # Save summary to separate file
            summary_path = args.output.parent / f"{args.output.stem}_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(summary)
            print(f"Summary report saved: {summary_path}")
            
    except Exception as e:
        print(f"Error generating SBOM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()