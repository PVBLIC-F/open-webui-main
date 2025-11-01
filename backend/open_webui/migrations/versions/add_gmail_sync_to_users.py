"""add gmail_sync_enabled to users

Revision ID: add_gmail_sync_to_users
Revises: 0b80d222da03
Create Date: 2024-11-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_gmail_sync_to_users'
down_revision = '0b80d222da03'
branch_labels = None
depends_on = None


def upgrade():
    # Add gmail_sync_enabled column to user table (PostgreSQL)
    op.add_column('user', sa.Column('gmail_sync_enabled', sa.Integer(), server_default='0', nullable=False))


def downgrade():
    # Remove gmail_sync_enabled column from user table (PostgreSQL)
    op.drop_column('user', 'gmail_sync_enabled')

