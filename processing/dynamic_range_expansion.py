import numpy as np


def dynamic_range_expansion(audio, sample_rate, threshold, ratio, attack, release):
    audio = np.copy(audio)
    expanded_audio = np.zeros_like(audio)
    gain = np.ones_like(audio)

    # Convert attack and release times to coefficients
    attack_coeff = np.exp(-1.0 / (attack * sample_rate))
    release_coeff = np.exp(-1.0 / (release * sample_rate))

    envelope = np.zeros_like(audio)
    for i in range(1, len(audio)):
        if np.abs(audio[i]) > envelope[i - 1]:
            envelope[i] = attack_coeff * envelope[i - 1] + (1 - attack_coeff) * np.abs(
                audio[i]
            )
        else:
            envelope[i] = release_coeff * envelope[i - 1] + (
                1 - release_coeff
            ) * np.abs(audio[i])

        if envelope[i] < threshold:
            gain[i] = 1 + (threshold - envelope[i]) * (ratio - 1) / threshold
        else:
            gain[i] = 1

        expanded_audio[i] = audio[i] * gain[i]

    return expanded_audio
