#!/usr/bin/env python

import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runstampede", action="store_true", default=False, help="run tests for issues on hold"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "stampede: mark test as only for on the Stampede cluster")


def pytest_collection_modifyitems(config, items):

    if not config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--runstampede"):
        # --runstampede given in cli: do not skip tests on hold
        skip_stampede = pytest.mark.skip(reason="need --runstampede option to run")
        for item in items:
            if "stampede" in item.keywords:
                item.add_marker(skip_stampede)
