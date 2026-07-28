"""Shared physical constants for the Model 2 (range controller) pipeline."""
import numpy as np

WHEEL_DIAMETER_M = 0.1651
WHEEL_CIRCUM_M = np.pi * WHEEL_DIAMETER_M

FULL_SPEED = 580  # raw hoverboard speed units at 100% (see dataset/run_scripts CONFIGS)

NOMINAL_CAP_AH = 10.2
NOMINAL_PACK_V = 42.0
NOMINAL_CAPACITY_WH = NOMINAL_CAP_AH * NOMINAL_PACK_V
