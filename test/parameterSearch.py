import os
import sys
import numpy as np

PATH = os.path.split(os.path.abspath(__file__))[0]
PARENT_PATH = os.path.dirname(PATH)
sys.path.append(PARENT_PATH)

from cross_validation import CV

