import pytest

from tests.testing import set_seed

__all__ = ["set_seed"]

SEED = 42


@pytest.fixture(autouse=True)
def seed_everything() -> None:
    set_seed(SEED)
