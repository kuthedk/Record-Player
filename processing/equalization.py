from scipy.signal import sosfilt, butter


def band_pass_filter(freq, Q, sample_rate):
    nyquist = sample_rate / 2
    low = max(0, (freq / nyquist) / (2 ** (1 / Q)))
    high = min(1, (freq / nyquist) * (2 ** (1 / Q)))
    sos = butter(2, [low, high], btype="band", output="sos")
    return sos


def equalize(audio, sample_rate):
    # Define equalizer settings: (frequency, Q, gain)
    eq_settings = [
        (100, 1, 0.8),  # Bass
        (250, 1, 0.9),  # Low mid
        (500, 1, 1.2),  # Mid
        (1000, 1, 1.0),  # Upper mid
        (4000, 1, 0.9),  # Presence
        (8000, 1, 0.8),  # Brilliance
    ]

    eq_audio = audio.copy()

    for freq, Q, gain in eq_settings:
        sos = band_pass_filter(freq, Q, sample_rate)
        filtered = sosfilt(sos, eq_audio)
        eq_audio += (filtered - eq_audio) * (gain - 1)

    return eq_audio
