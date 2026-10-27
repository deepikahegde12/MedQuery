import pytest

from tests.doubles import make_client


@pytest.fixture
def client():
    with make_client() as c:
        yield c
