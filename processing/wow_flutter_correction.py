from scipy.signal import resample
import numpy as np


def wow_flutter_correction(audio, sample_rate):
    t = np.arange(len(audio)) / sample_rate
    wow_flutter_effect = 1 + 0.01 * np.sin(2 * np.pi * 0.1 * t)

    corrected_audio = resample(audio, int(len(audio) / np.mean(wow_flutter_effect)))

    return corrected_audio
