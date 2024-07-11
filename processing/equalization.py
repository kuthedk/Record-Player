from scipy.signal import sosfilt, iirfilter


def equalize(audio, sample_rate):
    eq_curve = [
        (50, 0.8),
        (100, 0.6),
        (200, 0.8),
        (400, 1),
        (800, 1.2),
        (1600, 1),
        (3200, 0.8),
        (6400, 0.6),
        (12800, 0.8),
        (20000, 1),
    ]

    eq_curve_freqs, eq_curve_gain = zip(*eq_curve)
    nyquist = 0.5 * sample_rate
    sos = iirfilter(
        10,
        [f / nyquist for f in eq_curve_freqs if f / nyquist > 0],
        rp=5,
        rs=60,
        btype="band",
        ftype="cheby2",
        output="sos",
    )
    eq_audio = sosfilt(sos, audio)
    return eq_audio
