"""Add JSONB indexes for message table (channels optimization)

Revision ID: 80d73eec17ce
Revises: a5c220713937
Create Date: 2025-11-17 16:00:00.000000

Adds PostgreSQL JSONB optimizations for channel messages:
- Converts data and meta columns from JSON to JSONB
- Creates GIN indexes for fast JSON queries
- Optimizes for PostgreSQL 17 when available
- Safe for SQLite (no-op)
"""

from alembic import op
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
import logging

revision = "80d73eec17ce"
down_revision = "a5c220713937"
branch_labels = None
depends_on = None

log = logging.getLogger(__name__)


def _get_pg_version(conn) -> tuple[int, int]:
    """Get PostgreSQL major and minor version."""
    try:
        result = conn.execute(text("SHOW server_version_num"))
        version_num = int(result.scalar())
        major = version_num // 10000
        minor = (version_num // 100) % 100
        return (major, minor)
    except Exception as e:
        log.warning(f"Could not determine PostgreSQL version: {e}")
        return (0, 0)


def upgrade():
    """
    Add JSONB optimizations for message table.
    
    Converts JSON columns to JSONB and adds GIN indexes for:
    - message.data (files, attachments)
    - message.meta (sources, model info, reactions)
    """
    conn = op.get_bind()

    # Skip for non-PostgreSQL databases
    if conn.dialect.name != "postgresql":
        log.info(f"Skipping JSONB optimization for {conn.dialect.name}")
        return

    pg_major, pg_minor = _get_pg_version(conn)
    is_pg17_or_higher = pg_major >= 17
    log.info(f"PostgreSQL {pg_major}.{pg_minor} detected")

    try:
        # Convert JSON columns to JSONB for PostgreSQL
        log.info("Converting message.data from JSON to JSONB")
        conn.execute(
            text(
                "ALTER TABLE message "
                "ALTER COLUMN data TYPE JSONB USING data::jsonb"
            )
        )

        log.info("Converting message.meta from JSON to JSONB")
        conn.execute(
            text(
                "ALTER TABLE message "
                "ALTER COLUMN meta TYPE JSONB USING meta::jsonb"
            )
        )

        # Set statistics targets for better query planning
        conn.execute(text("ALTER TABLE message ALTER COLUMN data SET STATISTICS 1000"))
        conn.execute(text("ALTER TABLE message ALTER COLUMN meta SET STATISTICS 1000"))
        log.info("Set statistics targets for query optimization")

        # Create GIN indexes for JSONB columns
        # Index 1: message.data for file/knowledge queries
        log.info("Creating GIN index on message.data")
        conn.execute(
            text(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_message_data_gin "
                "ON message USING gin (data jsonb_path_ops) "
                "WITH (fastupdate = off)"
            )
        )

        # Index 2: message.meta for sources/model queries  
        log.info("Creating GIN index on message.meta")
        conn.execute(
            text(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_message_meta_gin "
                "ON message USING gin (meta jsonb_path_ops) "
                "WITH (fastupdate = off)"
            )
        )

        # Index 3: Optimized for meta.sources queries (knowledge integration)
        log.info("Creating GIN index on message.meta->'sources'")
        conn.execute(
            text(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_message_meta_sources_gin "
                "ON message USING gin ((meta->'sources') jsonb_path_ops) "
                "WITH (fastupdate = off)"
            )
        )

        # Composite index for channel + parent queries (thread performance)
        log.info("Creating composite index for thread queries")
        conn.execute(
            text(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_message_channel_parent "
                "ON message (channel_id, parent_id, created_at DESC)"
            )
        )

        # VACUUM ANALYZE for statistics update
        conn.execute(text("VACUUM ANALYZE message"))
        log.info("JSONB migration completed successfully")

    except ProgrammingError as e:
        if "already exists" in str(e) or "already JSONB" in str(e):
            log.info(f"JSONB already configured: {e}")
        else:
            log.error(f"JSONB migration error: {e}")
            raise


def downgrade():
    """Revert JSONB optimizations (convert back to JSON)"""
    conn = op.get_bind()

    if conn.dialect.name != "postgresql":
        return

    try:
        # Drop indexes
        conn.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_message_data_gin"))
        conn.execute(text("DROP INDEX CONCURRENTLY IF EXISTS idx_message_meta_gin"))
        conn.execute(
            text("DROP INDEX CONCURRENTLY IF EXISTS idx_message_meta_sources_gin")
        )
        conn.execute(
            text("DROP INDEX CONCURRENTLY IF EXISTS idx_message_channel_parent")
        )

        # Convert JSONB back to JSON
        conn.execute(
            text("ALTER TABLE message ALTER COLUMN data TYPE JSON USING data::json")
        )
        conn.execute(
            text("ALTER TABLE message ALTER COLUMN meta TYPE JSON USING meta::json")
        )

        log.info("JSONB indexes and types reverted")
    except Exception as e:
        log.error(f"Downgrade error: {e}")

