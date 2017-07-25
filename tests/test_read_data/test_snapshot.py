#!/usr/bin/env python
'''Testing for read_data.snapshot.py

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

from mock import call, patch
import numpy as np
import numpy.testing as npt
import os
import pdb
import pytest
import unittest

import galaxy_diver.read_data.snapshot as read_snapshot

sdir = './tests/test_data/test_analysis_dir'
sdir2 = './tests/test_data/test_analysis_dir2'
data_sdir = './tests/test_data/test_sdir'
data_sdir2 = './tests/test_data/test_sdir2'

########################################################################

# Decorator for skipping slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

########################################################################

