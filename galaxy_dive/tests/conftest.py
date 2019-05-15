#!/usr/bin/env python
'''Test suite config file.

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import pytest

########################################################################

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
        help="run slow tests")
    parser.addoption("--runstampede", action="store_true",
        help="run tests unique to stampede")
