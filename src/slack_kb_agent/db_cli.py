"""Database management CLI commands."""

import logging
import sys
from pathlib import Path

from .backup import BackupManager
from .database import (
    get_database_manager,
    get_database_repository,
    is_database_available,
)
from .persistent_knowledge_base import PersistentKnowledgeBase

logger = logging.getLogger(__name__)


def init_database() -> int:
    """Initialize database schema."""
    try:
        db_manager = get_database_manager()
        db_manager.initialize()
        print("✅ Database schema initialized successfully")
        return 0
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return 1


def check_database() -> int:
    """Check database connectivity and status."""
    try:
        if not is_database_available():
            print("❌ Database is not available")
            return 1

        db_repo = get_database_repository()
        stats = db_repo.get_memory_stats()

        print("✅ Database is available")
        print(f"📊 Total documents: {stats.get('total_documents', 0)}")
        print(f"💾 Estimated size: {stats.get('estimated_size_bytes', 0)} bytes")
        print(f"🔗 Database: {stats.get('database_url', 'unknown')}")

        if stats.get('source_distribution'):
            print("📁 Documents by source:")
            for source, count in stats['source_distribution'].items():
                print(f"  - {source}: {count} documents")

        return 0
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return 1


def migrate_json_to_database(json_path: str) -> int:
    """Migrate documents from JSON file to database."""
    try:
        json_file = Path(json_path)
        if not json_file.exists():
            print(f"❌ JSON file not found: {json_path}")
            return 1

        if not is_database_available():
            print("❌ Database is not available")
            return 1

        print(f"📥 Loading documents from {json_path}...")

        # Load using PersistentKnowledgeBase (which auto-syncs to database)
        kb = PersistentKnowledgeBase.load(json_path, use_database=True)

        # Get final stats
        stats = kb.get_database_stats()
        total_docs = stats.get('total_documents', 0)

        print(f"✅ Migration completed: {total_docs} documents migrated to database")
        return 0

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1


def export_database_to_json(json_path: str) -> int:
    """Export all database documents to JSON file."""
    try:
        if not is_database_available():
            print("❌ Database is not available")
            return 1

        print("📤 Exporting documents from database...")

        # Load from database and save to JSON
        kb = PersistentKnowledgeBase.load_from_database()
        kb.save(json_path)

        print(f"✅ Export completed: {len(kb.documents)} documents exported to {json_path}")
        return 0

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1


def backup_database(backup_path: str, compress: bool = True) -> int:
    """Create a backup of the database."""
    try:
        if not is_database_available():
            print("❌ Database is not available")
            return 1

        print("💾 Creating database backup...")

        backup_manager = BackupManager()
        result = backup_manager.create_backup(backup_path, compress=compress)

        print("✅ Backup completed:")
        print(f"  - File: {result['backup_path']}")
        print(f"  - Documents: {result['document_count']}")
        print(f"  - Size: {result['file_size_bytes']} bytes")
        print(f"  - Compressed: {result['compressed']}")

        return 0

    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return 1


def restore_database(backup_path: str, clear_existing: bool = False) -> int:
    """Restore database from backup."""
    try:
        backup_file = Path(backup_path)
        if not backup_file.exists():
            print(f"❌ Backup file not found: {backup_path}")
            return 1

        if not is_database_available():
            print("❌ Database is not available")
            return 1

        print(f"📥 Restoring database from {backup_path}...")
        if clear_existing:
            print("⚠️  Existing documents will be cleared!")

        backup_manager = BackupManager()
        result = backup_manager.restore_backup(backup_path, clear_existing=clear_existing)

        print("✅ Restore completed:")
        print(f"  - Documents restored: {result['documents_restored']}")
        if result.get('documents_cleared', 0) > 0:
            print(f"  - Documents cleared: {result['documents_cleared']}")

        return 0

    except Exception as e:
        print(f"❌ Restore failed: {e}")
        return 1


def validate_backup(backup_path: str) -> int:
    """Validate a backup file."""
    try:
        backup_file = Path(backup_path)
        if not backup_file.exists():
            print(f"❌ Backup file not found: {backup_path}")
            return 1

        print(f"🔍 Validating backup file {backup_path}...")

        backup_manager = BackupManager()
        result = backup_manager.restore_backup(backup_path, validate_only=True)

        validation = result['validation']

        if validation['valid']:
            print("✅ Backup file is valid")
            print(f"  - Documents: {result['document_count']}")

            backup_meta = result.get('backup_metadata', {})
            if backup_meta.get('created_at'):
                print(f"  - Created: {backup_meta['created_at']}")
            if backup_meta.get('version'):
                print(f"  - Version: {backup_meta['version']}")
        else:
            print("❌ Backup file is invalid")
            for error in validation['errors']:
                print(f"  - Error: {error}")

        if validation['warnings']:
            print("⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        return 0 if validation['valid'] else 1

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1


def clear_database() -> int:
    """Clear all documents from database."""
    try:
        if not is_database_available():
            print("❌ Database is not available")
            return 1

        # Confirm with user
        response = input("⚠️  This will delete ALL documents from the database. Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled")
            return 1

        db_repo = get_database_repository()
        cleared_count = db_repo.clear_all_documents()

        print(f"✅ Cleared {cleared_count} documents from database")
        return 0

    except Exception as e:
        print(f"❌ Clear operation failed: {e}")
        return 1


def main():
    """Database management CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Slack KB Agent Database Management")
    subparsers = parser.add_subparsers(dest='command', help='Database commands')

    # Init command
    subparsers.add_parser('init', help='Initialize database schema')

    # Check command
    subparsers.add_parser('check', help='Check database status')

    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate JSON file to database')
    migrate_parser.add_argument('json_path', help='Path to JSON file')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export database to JSON file')
    export_parser.add_argument('json_path', help='Output JSON file path')

    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create database backup')
    backup_parser.add_argument('backup_path', help='Backup file path')
    backup_parser.add_argument('--no-compress', action='store_true', help='Disable compression')

    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore database from backup')
    restore_parser.add_argument('backup_path', help='Backup file path')
    restore_parser.add_argument('--clear', action='store_true', help='Clear existing documents')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate backup file')
    validate_parser.add_argument('backup_path', help='Backup file path')

    # Clear command
    subparsers.add_parser('clear', help='Clear all documents from database')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Configure basic logging
    logging.basicConfig(level=logging.WARNING)

    try:
        if args.command == 'init':
            return init_database()
        elif args.command == 'check':
            return check_database()
        elif args.command == 'migrate':
            return migrate_json_to_database(args.json_path)
        elif args.command == 'export':
            return export_database_to_json(args.json_path)
        elif args.command == 'backup':
            return backup_database(args.backup_path, compress=not args.no_compress)
        elif args.command == 'restore':
            return restore_database(args.backup_path, clear_existing=args.clear)
        elif args.command == 'validate':
            return validate_backup(args.backup_path)
        elif args.command == 'clear':
            return clear_database()
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1


if __name__ == '__main__':
    sys.exit(main())
