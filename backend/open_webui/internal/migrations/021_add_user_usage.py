"""Peewee migrations -- 019_add_user_usage.py.

Adds user_usage table for tracking token usage and costs per user.
Also adds spend limit fields to the user table.

Some examples (model - class or model name)::

    > Model = migrator.orm['table_name']            # Return model in current state by name
    > Model = migrator.ModelClass                   # Return model in current state by name

    > migrator.sql(sql)                             # Run custom SQL
    > migrator.run(func, *args, **kwargs)           # Run python function with the given args
    > migrator.create_model(Model)                  # Create a model (could be used as decorator)
    > migrator.remove_model(model, cascade=True)    # Remove a model
    > migrator.add_fields(model, **fields)          # Add fields to a model
    > migrator.change_fields(model, **fields)       # Change fields
    > migrator.remove_fields(model, *field_names, cascade=True)
    > migrator.rename_field(model, old_field_name, new_field_name)
    > migrator.rename_table(model, new_table_name)
    > migrator.add_index(model, *col_names, unique=False)
    > migrator.add_not_null(model, *field_names)
    > migrator.add_default(model, field_name, default)
    > migrator.add_constraint(model, name, sql)
    > migrator.drop_index(model, *col_names)
    > migrator.drop_not_null(model, *field_names)
    > migrator.drop_constraints(model, *constraints)

"""

from contextlib import suppress

import peewee as pw
from peewee_migrate import Migrator


with suppress(ImportError):
    import playhouse.postgres_ext as pw_pext


def migrate(migrator: Migrator, database: pw.Database, *, fake=False):
    """Write your migrations here."""

    # Create user_usage table for tracking token usage and costs
    @migrator.create_model
    class UserUsage(pw.Model):
        id = pw.AutoField()
        user_id = pw.TextField(index=True)
        date = pw.DateField(index=True)
        model_id = pw.TextField()
        input_tokens = pw.BigIntegerField(default=0)
        output_tokens = pw.BigIntegerField(default=0)
        reasoning_tokens = pw.BigIntegerField(default=0)
        total_tokens = pw.BigIntegerField(default=0)
        cost = pw.DoubleField(default=0.0)
        request_count = pw.IntegerField(default=0)
        created_at = pw.BigIntegerField()
        updated_at = pw.BigIntegerField()

        class Meta:
            table_name = "user_usage"
            indexes = (
                # Composite index for efficient queries
                (("user_id", "date"), False),
                # Unique constraint for upsert operations
                (("user_id", "date", "model_id"), True),
            )

    # Add spend limit fields to user table
    migrator.add_fields(
        "user",
        # Spend limits (in USD) - null means no limit
        spend_limit_daily=pw.DoubleField(null=True, default=None),
        spend_limit_monthly=pw.DoubleField(null=True, default=None),
        # Whether spend limits are enabled for this user
        spend_limit_enabled=pw.BooleanField(default=False),
    )


def rollback(migrator: Migrator, database: pw.Database, *, fake=False):
    """Write your rollback migrations here."""

    migrator.remove_model("user_usage")
    migrator.remove_fields(
        "user",
        "spend_limit_daily",
        "spend_limit_monthly",
        "spend_limit_enabled",
    )
