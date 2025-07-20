"""Knowledge source ingestion system for populating the knowledge base."""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse

from .knowledge_base import KnowledgeBase
from .models import Document

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import requests
    from bs4 import BeautifulSoup
    import markdown
    from slack_sdk import WebClient
    INGESTION_DEPS_AVAILABLE = True
except ImportError:
    INGESTION_DEPS_AVAILABLE = False
    requests = None
    BeautifulSoup = None
    markdown = None
    WebClient = None


class BaseIngester(ABC):
    """Abstract base class for all ingestion sources."""
    
    @abstractmethod
    def ingest(self, **kwargs) -> List[Document]:
        """Ingest documents from the source."""
        raise NotImplementedError


class ContentProcessor:
    """Process and filter content during ingestion."""
    
    def __init__(self):
        # Patterns for sensitive content detection
        self.sensitive_patterns = [
            r'api[_\-\s]*key[_\-\s]*[:=]\s*["\']?([a-zA-Z0-9_\-]{10,})["\']?',
            r'password[_\-\s]*[:=]\s*["\']?([^\s"\']{6,})["\']?',
            r'token[_\-\s]*[:=]\s*["\']?([a-zA-Z0-9_\-]{15,})["\']?',
            r'secret[_\-\s]*[:=]\s*["\']?([a-zA-Z0-9_\-]{10,})["\']?',
            r'(sk-[a-zA-Z0-9\-]{10,})',  # OpenAI-style API keys
            r'(xox[bpoas]-[a-zA-Z0-9\-]{10,})',  # Slack tokens
        ]
    
    def process_markdown(self, content: str) -> str:
        """Convert markdown to plain text while preserving structure."""
        if not INGESTION_DEPS_AVAILABLE:
            return content
        
        try:
            # Convert markdown to HTML then to text
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            logger.warning(f"Failed to process markdown: {e}")
            return content
    
    def filter_sensitive_content(self, content: str) -> str:
        """Remove sensitive information from content."""
        filtered = content
        
        for pattern in self.sensitive_patterns:
            filtered = re.sub(pattern, '[REDACTED_SENSITIVE_DATA]', filtered, flags=re.IGNORECASE)
        
        return filtered
    
    def clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Filter sensitive content
        content = self.filter_sensitive_content(content)
        
        return content.strip()


class FileIngester(BaseIngester):
    """Ingest documents from local files and directories."""
    
    def __init__(self, processor: Optional[ContentProcessor] = None):
        self.processor = processor or ContentProcessor()
        self.supported_extensions = {'.md', '.txt', '.rst', '.py', '.js', '.ts', '.json', '.yaml', '.yml'}
    
    def ingest_directory(self, path: Path, recursive: bool = True) -> List[Document]:
        """Ingest all supported files from a directory."""
        documents = []
        
        try:
            path = Path(path)
            if not path.exists():
                logger.error(f"Directory does not exist: {path}")
                return documents
            
            pattern = "**/*" if recursive else "*"
            for file_path in path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    doc = self.ingest_file(file_path)
                    if doc:
                        documents.append(doc)
        
        except Exception as e:
            logger.error(f"Error ingesting directory {path}: {e}")
        
        return documents
    
    def ingest_file(self, file_path: Path) -> Optional[Document]:
        """Ingest a single file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Process based on file type
            if file_path.suffix.lower() == '.md':
                content = self.processor.process_markdown(content)
            
            content = self.processor.clean_content(content)
            
            if not content.strip():
                return None
            
            return Document(
                content=content,
                source=f"file:{file_path.name}",
                metadata={
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "extension": file_path.suffix
                }
            )
        
        except Exception as e:
            logger.warning(f"Failed to ingest file {file_path}: {e}")
            return None
    
    def ingest(self, path: str, recursive: bool = True) -> List[Document]:
        """Ingest from path (file or directory)."""
        path_obj = Path(path)
        
        if path_obj.is_file():
            doc = self.ingest_file(path_obj)
            return [doc] if doc else []
        elif path_obj.is_dir():
            return self.ingest_directory(path_obj, recursive)
        else:
            logger.error(f"Path does not exist: {path}")
            return []


class GitHubIngester(BaseIngester):
    """Ingest documents from GitHub repositories."""
    
    def __init__(self, token: Optional[str] = None, processor: Optional[ContentProcessor] = None):
        self.token = token
        self.processor = processor or ContentProcessor()
        
        if not INGESTION_DEPS_AVAILABLE:
            raise ImportError("GitHub ingestion requires 'requests' package")
    
    def ingest_repository(self, repo: str, include_issues: bool = True, include_readme: bool = True) -> List[Document]:
        """Ingest content from a GitHub repository."""
        documents = []
        
        try:
            if include_issues:
                documents.extend(self._ingest_issues(repo))
            
            if include_readme:
                documents.extend(self._ingest_readme(repo))
        
        except Exception as e:
            logger.error(f"Error ingesting GitHub repository {repo}: {e}")
        
        return documents
    
    def _ingest_issues(self, repo: str) -> List[Document]:
        """Ingest GitHub issues and pull requests."""
        documents = []
        
        headers = {"Authorization": f"token {self.token}"} if self.token else {}
        
        try:
            # Get issues (includes PRs)
            url = f"https://api.github.com/repos/{repo}/issues"
            response = requests.get(url, headers=headers, params={"state": "all", "per_page": 100}, timeout=30)
            response.raise_for_status()
            
            for issue in response.json():
                title = issue.get("title", "")
                body = issue.get("body", "") or ""
                number = issue.get("number", 0)
                state = issue.get("state", "unknown")
                labels = [label.get("name", "") for label in issue.get("labels", [])]
                
                # Combine title and body
                content = f"# {title}\n\n{body}"
                content = self.processor.clean_content(content)
                
                if content.strip():
                    doc = Document(
                        content=content,
                        source=f"github:{repo}",
                        metadata={
                            "type": "pull_request" if "pull_request" in issue else "issue",
                            "number": number,
                            "state": state,
                            "labels": labels,
                            "url": issue.get("html_url", "")
                        }
                    )
                    documents.append(doc)
        
        except Exception as e:
            logger.error(f"Failed to ingest GitHub issues for {repo}: {e}")
        
        return documents
    
    def _ingest_readme(self, repo: str) -> List[Document]:
        """Ingest repository README file."""
        documents = []
        
        headers = {"Authorization": f"token {self.token}"} if self.token else {}
        
        try:
            url = f"https://api.github.com/repos/{repo}/readme"
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                readme_data = response.json()
                
                # Get the actual content
                content_url = readme_data.get("download_url")
                if content_url:
                    content_response = requests.get(content_url, timeout=30)
                    content_response.raise_for_status()
                    
                    content = content_response.text
                    content = self.processor.process_markdown(content)
                    content = self.processor.clean_content(content)
                    
                    if content.strip():
                        doc = Document(
                            content=content,
                            source=f"github:{repo}",
                            metadata={
                                "type": "readme",
                                "filename": readme_data.get("name", "README"),
                                "path": readme_data.get("path", ""),
                                "url": readme_data.get("html_url", "")
                            }
                        )
                        documents.append(doc)
        
        except Exception as e:
            logger.warning(f"Failed to ingest README for {repo}: {e}")
        
        return documents
    
    def ingest(self, repo: str, **kwargs) -> List[Document]:
        """Ingest from GitHub repository."""
        return self.ingest_repository(repo, **kwargs)


class WebDocumentationCrawler(BaseIngester):
    """Crawl and ingest web documentation."""
    
    def __init__(self, processor: Optional[ContentProcessor] = None):
        self.processor = processor or ContentProcessor()
        self.visited_urls: Set[str] = set()
        
        if not INGESTION_DEPS_AVAILABLE:
            raise ImportError("Web crawling requires 'requests' and 'beautifulsoup4' packages")
    
    def crawl_url(self, url: str, max_depth: int = 2) -> List[Document]:
        """Crawl a documentation website."""
        documents = []
        self.visited_urls.clear()
        
        try:
            documents.extend(self._crawl_recursive(url, max_depth, 0))
        except Exception as e:
            logger.error(f"Error crawling URL {url}: {e}")
        
        return documents
    
    def _crawl_recursive(self, url: str, max_depth: int, current_depth: int) -> List[Document]:
        """Recursively crawl pages."""
        documents = []
        
        if current_depth > max_depth or url in self.visited_urls:
            return documents
        
        self.visited_urls.add(url)
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            if "text/html" not in response.headers.get("content-type", ""):
                return documents
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            content = self._extract_content(soup)
            content = self.processor.clean_content(content)
            
            if content.strip():
                doc = Document(
                    content=content,
                    source=f"web:{urlparse(url).netloc}",
                    metadata={
                        "url": url,
                        "title": soup.title.string if soup.title else "",
                        "depth": current_depth
                    }
                )
                documents.append(doc)
            
            # Find links to crawl next (only same domain)
            if current_depth < max_depth:
                base_domain = urlparse(url).netloc
                
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(url, link['href'])
                    next_domain = urlparse(next_url).netloc
                    
                    if next_domain == base_domain and next_url not in self.visited_urls:
                        documents.extend(self._crawl_recursive(next_url, max_depth, current_depth + 1))
        
        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {e}")
        
        return documents
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract meaningful content from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Look for main content areas
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', class_=re.compile(r'content|main|article', re.I)) or
            soup.find('body')
        )
        
        if main_content:
            return main_content.get_text(separator='\n', strip=True)
        else:
            return soup.get_text(separator='\n', strip=True)
    
    def ingest(self, url: str, max_depth: int = 2) -> List[Document]:
        """Ingest from web documentation."""
        return self.crawl_url(url, max_depth)


class SlackHistoryIngester(BaseIngester):
    """Ingest Slack conversation history."""
    
    def __init__(self, token: str, processor: Optional[ContentProcessor] = None):
        self.token = token
        self.processor = processor or ContentProcessor()
        
        if not INGESTION_DEPS_AVAILABLE:
            raise ImportError("Slack ingestion requires 'slack-sdk' package")
        
        self.client = WebClient(token=token)
    
    def ingest_channel(self, channel: str, days: int = 30) -> List[Document]:
        """Ingest messages from a Slack channel."""
        documents = []
        
        try:
            # Calculate timestamp for X days ago
            oldest = time.time() - (days * 24 * 60 * 60)
            
            response = self.client.conversations_history(
                channel=channel,
                oldest=oldest,
                limit=1000
            )
            
            for message in response["messages"]:
                text = message.get("text", "")
                user = message.get("user", "unknown")
                ts = message.get("ts", "")
                
                if text.strip():
                    content = self.processor.clean_content(text)
                    
                    doc = Document(
                        content=content,
                        source=f"slack:{channel}",
                        metadata={
                            "user": user,
                            "timestamp": ts,
                            "channel": channel,
                            "type": "message"
                        }
                    )
                    documents.append(doc)
        
        except Exception as e:
            logger.error(f"Failed to ingest Slack channel {channel}: {e}")
        
        return documents
    
    def ingest(self, channel: str, days: int = 30) -> List[Document]:
        """Ingest from Slack channel."""
        return self.ingest_channel(channel, days)


class IngestionTracker:
    """Track ingested content to avoid duplicates."""
    
    def __init__(self):
        self.ingested_checksums: Dict[str, str] = {}
    
    def add_document(self, document: Document, checksum: str) -> None:
        """Add a document to the tracking system."""
        self.ingested_checksums[self._get_doc_key(document)] = checksum
    
    def should_ingest_document(self, source_id: str, checksum: str) -> bool:
        """Check if a document should be ingested."""
        return self.ingested_checksums.get(source_id) != checksum
    
    def _get_doc_key(self, document: Document) -> str:
        """Generate a unique key for a document."""
        if "path" in document.metadata:
            return document.metadata["path"]
        elif "url" in document.metadata:
            return document.metadata["url"]
        else:
            return f"{document.source}:{hashlib.md5(document.content.encode()).hexdigest()[:8]}"


class BatchIngester:
    """Handle batch ingestion for performance."""
    
    def __init__(self, knowledge_base: KnowledgeBase, batch_size: int = 50):
        self.knowledge_base = knowledge_base
        self.batch_size = batch_size
    
    def ingest_batch(self, documents: List[Document]) -> None:
        """Ingest documents in batches."""
        logger.info(f"Ingesting {len(documents)} documents in batches of {self.batch_size}")
        
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            self.knowledge_base.add_documents(batch)
            
            logger.info(f"Ingested batch {i // self.batch_size + 1} ({len(batch)} documents)")


class IngestionProgress:
    """Track progress of ingestion operations."""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
    
    def update(self, processed: int) -> None:
        """Update progress."""
        self.processed_items = processed
    
    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if ingestion is complete."""
        return self.processed_items >= self.total_items
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time


def main():
    """Command line interface for ingestion."""
    parser = argparse.ArgumentParser(description="Ingest knowledge sources")
    parser.add_argument("--source", required=True, choices=["files", "github", "web", "slack"])
    parser.add_argument("--output", default="kb.json", help="Output knowledge base file")
    
    # Source-specific arguments
    parser.add_argument("--path", help="Path for file ingestion")
    parser.add_argument("--repo", help="GitHub repository (owner/name)")
    parser.add_argument("--url", help="Web documentation URL")
    parser.add_argument("--channel", help="Slack channel")
    parser.add_argument("--token", help="API token (GitHub/Slack)")
    parser.add_argument("--recursive", action="store_true", help="Recursive file ingestion")
    parser.add_argument("--days", type=int, default=30, help="Days of history for Slack")
    
    args = parser.parse_args()
    
    # Initialize knowledge base
    kb = KnowledgeBase()
    
    # Try to load existing knowledge base
    kb_path = Path(args.output)
    if kb_path.exists():
        kb = KnowledgeBase.load(kb_path)
        print(f"Loaded existing knowledge base with {len(kb.documents)} documents")
    
    # Initialize ingester based on source
    documents = []
    
    try:
        if args.source == "files":
            if not args.path:
                raise ValueError("--path required for file ingestion")
            ingester = FileIngester()
            documents = ingester.ingest(args.path, recursive=args.recursive)
        
        elif args.source == "github":
            if not args.repo:
                raise ValueError("--repo required for GitHub ingestion")
            ingester = GitHubIngester(token=args.token)
            documents = ingester.ingest(args.repo)
        
        elif args.source == "web":
            if not args.url:
                raise ValueError("--url required for web ingestion")
            ingester = WebDocumentationCrawler()
            documents = ingester.ingest(args.url)
        
        elif args.source == "slack":
            if not args.channel or not args.token:
                raise ValueError("--channel and --token required for Slack ingestion")
            ingester = SlackHistoryIngester(token=args.token)
            documents = ingester.ingest(args.channel, days=args.days)
        
        # Batch ingest documents
        if documents:
            batch_ingester = BatchIngester(kb)
            batch_ingester.ingest_batch(documents)
            
            print(f"Ingested {len(documents)} new documents")
            print(f"Total documents in knowledge base: {len(kb.documents)}")
            
            # Save knowledge base
            kb.save(kb_path)
            print(f"Saved knowledge base to {kb_path}")
        else:
            print("No documents were ingested")
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit(main())