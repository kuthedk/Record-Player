import numpy as np
from scipy.signal import correlate


def phase_correction(left_channel, right_channel):
    corr = correlate(left_channel, right_channel, mode="full")
    delay = np.argmax(corr) - len(right_channel) + 1

    if delay > 0:
        corrected_left = np.pad(left_channel, (delay, 0), "constant")[
            : len(left_channel)
        ]
        corrected_right = right_channel
    else:
        corrected_left = left_channel
        corrected_right = np.pad(right_channel, (abs(delay), 0), "constant")[
            : len(right_channel)
        ]

    return corrected_left, corrected_right
