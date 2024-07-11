import numpy as np


def noise_reduction(audio, sample_rate, noise_profile_duration=0.5):
    noise_profile_samples = int(noise_profile_duration * sample_rate)
    noise_profile = audio[:noise_profile_samples]

    noise_threshold = np.mean(np.abs(noise_profile)) * 1.5
    reduced_noise_audio = np.copy(audio)
    noise_mask = np.abs(audio) < noise_threshold
    reduced_noise_audio[noise_mask] = 0
    return reduced_noise_audio
