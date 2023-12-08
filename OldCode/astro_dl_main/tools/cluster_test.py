#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.python.client import device_lib
print("hello world")

print(np.arange(10).sum())
np.arange(10).sum()

print(device_lib.list_local_devices())
