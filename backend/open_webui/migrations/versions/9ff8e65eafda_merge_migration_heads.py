"""merge migration heads

Revision ID: 9ff8e65eafda
Revises: 544ba9a2b077, a5c220713937
Create Date: 2025-10-07 04:17:49.588653

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import open_webui.internal.db


# revision identifiers, used by Alembic.
revision: str = '9ff8e65eafda'
down_revision: Union[str, None] = ('544ba9a2b077', 'a5c220713937')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
