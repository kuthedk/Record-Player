from scipy.signal import butter, lfilter
import numpy as np


def sidechain_de_esser(audio, sample_rate, sibilance_threshold, ratio, attack, release):
    sibilance_freq_range = [5000, 10000]
    nyquist = 0.5 * sample_rate
    low = sibilance_freq_range[0] / nyquist
    high = sibilance_freq_range[1] / nyquist
    if low > 0 and high > 0:
        b, a = butter(4, [low, high], btype="band")
        sibilance = lfilter(b, a, audio)
    else:
        sibilance = audio

    gain_reduction = np.zeros_like(sibilance)
    envelope = np.zeros_like(sibilance)

    attack_coeff = np.exp(-1.0 / (attack * sample_rate))
    release_coeff = np.exp(-1.0 / (release * sample_rate))

    for i in range(1, len(sibilance)):
        if np.abs(sibilance[i]) > envelope[i - 1]:
            envelope[i] = attack_coeff * envelope[i - 1] + (1 - attack_coeff) * np.abs(
                sibilance[i]
            )
        else:
            envelope[i] = release_coeff * envelope[i - 1] + (
                1 - release_coeff
            ) * np.abs(sibilance[i])

        gain_reduction[i] = np.maximum(
            0, 1 - sibilance_threshold / (envelope[i] + 1e-6)
        )
        gain_reduction[i] = np.minimum(gain_reduction[i], 1.0 / ratio)

    compressed_audio = audio * (1 - gain_reduction)
    return compressed_audio
