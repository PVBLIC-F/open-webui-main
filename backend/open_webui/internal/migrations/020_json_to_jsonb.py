"""Peewee migrations -- 020_json_to_jsonb.py.

Placeholder migration - the actual migration was applied previously but the file was missing.
This is a no-op to satisfy the migration system.
"""

from contextlib import suppress

import peewee as pw
from peewee_migrate import Migrator


with suppress(ImportError):
    import playhouse.postgres_ext as pw_pext


def migrate(migrator: Migrator, database: pw.Database, *, fake=False):
    """No-op - migration was previously applied."""
    pass


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """No-op - migration was previously applied."""
    pass
