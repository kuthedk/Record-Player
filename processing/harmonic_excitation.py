import numpy as np


def harmonic_excitation(audio, sample_rate, amount=0.2):
    harmonic_audio = audio + amount * np.tanh(audio)
    return harmonic_audio
