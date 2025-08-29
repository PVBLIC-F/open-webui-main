"""merge migration heads

Revision ID: c8b3efb60779
Revises: 3af16a1c9fb6, 544ba9a2b077
Create Date: 2025-08-29 08:55:42.058750

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import open_webui.internal.db


# revision identifiers, used by Alembic.
revision: str = 'c8b3efb60779'
down_revision: Union[str, None] = ('3af16a1c9fb6', '544ba9a2b077')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
