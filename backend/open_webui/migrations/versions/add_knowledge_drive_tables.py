"""Add knowledge drive source and file tables

Revision ID: a1b2c3d4e5f6
Revises: 0b80d222da03
Create Date: 2026-01-13 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6"
down_revision = "0b80d222da03"
branch_labels = None
depends_on = None


def upgrade():
    # Create knowledge_drive_source table
    op.create_table(
        "knowledge_drive_source",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("knowledge_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("drive_folder_id", sa.Text(), nullable=False),
        sa.Column("drive_folder_name", sa.Text(), nullable=True),
        sa.Column("drive_folder_path", sa.Text(), nullable=True),
        sa.Column("last_sync_timestamp", sa.BigInteger(), nullable=True),
        sa.Column("last_sync_change_token", sa.Text(), nullable=True),
        sa.Column("last_sync_file_count", sa.Integer(), nullable=False, default=0),
        sa.Column("sync_status", sa.String(), nullable=False, default="never"),
        sa.Column("sync_enabled", sa.Boolean(), nullable=False, default=True),
        sa.Column("auto_sync_interval_hours", sa.Integer(), nullable=False, default=1),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("error_count", sa.Integer(), nullable=False, default=0),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for knowledge_drive_source
    op.create_index(
        "idx_knowledge_drive_source_knowledge_id",
        "knowledge_drive_source",
        ["knowledge_id"],
    )
    op.create_index(
        "idx_knowledge_drive_source_user_id",
        "knowledge_drive_source",
        ["user_id"],
    )
    op.create_index(
        "idx_knowledge_drive_source_sync_status",
        "knowledge_drive_source",
        ["sync_status"],
    )

    # Create knowledge_drive_file table
    op.create_table(
        "knowledge_drive_file",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("drive_source_id", sa.Text(), nullable=False),
        sa.Column("knowledge_id", sa.Text(), nullable=False),
        sa.Column("file_id", sa.Text(), nullable=True),  # FK to file.id
        sa.Column("drive_file_id", sa.Text(), nullable=False),
        sa.Column("drive_file_name", sa.Text(), nullable=False),
        sa.Column("drive_file_mime_type", sa.Text(), nullable=True),
        sa.Column("drive_file_size", sa.BigInteger(), nullable=True),
        sa.Column("drive_file_md5", sa.Text(), nullable=True),
        sa.Column("drive_file_modified_time", sa.Text(), nullable=True),
        sa.Column("last_sync_timestamp", sa.BigInteger(), nullable=True),
        sa.Column("sync_status", sa.String(), nullable=False, default="pending"),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for knowledge_drive_file
    op.create_index(
        "idx_knowledge_drive_file_source_id",
        "knowledge_drive_file",
        ["drive_source_id"],
    )
    op.create_index(
        "idx_knowledge_drive_file_knowledge_id",
        "knowledge_drive_file",
        ["knowledge_id"],
    )
    op.create_index(
        "idx_knowledge_drive_file_drive_id",
        "knowledge_drive_file",
        ["drive_file_id"],
    )


def downgrade():
    # Drop indexes for knowledge_drive_file
    op.drop_index("idx_knowledge_drive_file_drive_id", table_name="knowledge_drive_file")
    op.drop_index("idx_knowledge_drive_file_knowledge_id", table_name="knowledge_drive_file")
    op.drop_index("idx_knowledge_drive_file_source_id", table_name="knowledge_drive_file")

    # Drop knowledge_drive_file table
    op.drop_table("knowledge_drive_file")

    # Drop indexes for knowledge_drive_source
    op.drop_index("idx_knowledge_drive_source_sync_status", table_name="knowledge_drive_source")
    op.drop_index("idx_knowledge_drive_source_user_id", table_name="knowledge_drive_source")
    op.drop_index("idx_knowledge_drive_source_knowledge_id", table_name="knowledge_drive_source")

    # Drop knowledge_drive_source table
    op.drop_table("knowledge_drive_source")
