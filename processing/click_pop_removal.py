import numpy as np
from scipy.signal import find_peaks, butter, lfilter


def remove_clicks_pops(audio, sample_rate):
    peaks, _ = find_peaks(np.abs(audio), height=np.mean(np.abs(audio)) * 2)
    mask = np.ones_like(audio, dtype=bool)
    mask[peaks] = False
    interpolated_audio = np.interp(
        np.arange(len(audio)), np.arange(len(audio))[mask], audio[mask]
    )

    # Smoothing the interpolated regions
    lowcut = 1000
    nyquist = 0.5 * sample_rate
    if lowcut / nyquist > 0:
        b, a = butter(2, lowcut / nyquist, btype="low")
        smoothed_audio = lfilter(b, a, interpolated_audio)
        return smoothed_audio
    return interpolated_audio
