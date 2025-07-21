"""Initial schema for documents table

Revision ID: 0001
Revises: 
Create Date: 2025-07-21 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create documents table
    op.create_table('documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('source', sa.String(length=255), nullable=False),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_documents_source', 'documents', ['source'], unique=False)
    op.create_index('idx_documents_created_at', 'documents', ['created_at'], unique=False)
    op.create_index('idx_documents_content_search', 'documents', ['content'], unique=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_documents_content_search', table_name='documents')
    op.drop_index('idx_documents_created_at', table_name='documents')
    op.drop_index('idx_documents_source', table_name='documents')
    
    # Drop table
    op.drop_table('documents')