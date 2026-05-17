"""Smoke test — verify async test infra works."""
import asyncio


async def test_async_basic():
    await asyncio.sleep(0)
    assert True


def test_sync_basic():
    assert 1 + 1 == 2
