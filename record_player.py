import pyaudio
import numpy as np
from scipy import signal


def list_audio_devices(pyaudio_instance: pyaudio.PyAudio) -> None:
    print("Available audio devices:")
    for device_index in range(pyaudio_instance.get_device_count()):
        device_info = pyaudio_instance.get_device_info_by_index(device_index)
        print(f"Device {device_index}: {device_info['name']}")
        print(f"  Max Input Channels: {device_info['maxInputChannels']}")
        print(f"  Max Output Channels: {device_info['maxOutputChannels']}")
        print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")


def get_device_index(prompt: str) -> int:
    return int(input(prompt))


def simple_stereo_enhance(
    left_channel: np.ndarray, right_channel: np.ndarray, width: float = 1.8
) -> tuple[np.ndarray, np.ndarray]:
    mid_channel = (left_channel + right_channel) / 2
    side_channel = (left_channel - right_channel) / 2
    enhanced_left = mid_channel + width * side_channel
    enhanced_right = mid_channel - width * side_channel
    return enhanced_left, enhanced_right


def adaptive_de_esser(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    # Define frequency bands
    low_band = (20, 3000)
    mid_band = (3000, 6000)
    high_band = (6000, 20000)

    # Design band-pass filters
    nyquist = sample_rate / 2
    low_sos = signal.butter(
        10,
        [low_band[0] / nyquist, low_band[1] / nyquist],
        btype="bandpass",
        output="sos",
    )
    mid_sos = signal.butter(
        10,
        [mid_band[0] / nyquist, mid_band[1] / nyquist],
        btype="bandpass",
        output="sos",
    )
    high_sos = signal.butter(
        10,
        [high_band[0] / nyquist, high_band[1] / nyquist],
        btype="bandpass",
        output="sos",
    )

    # Apply filters
    low_band_audio = signal.sosfilt(low_sos, audio)
    mid_band_audio = signal.sosfilt(mid_sos, audio)
    high_band_audio = signal.sosfilt(high_sos, audio)

    # Compute envelopes
    def compute_envelope(x):
        return np.abs(signal.hilbert(x))

    low_env = compute_envelope(low_band_audio)
    mid_env = compute_envelope(mid_band_audio)
    high_env = compute_envelope(high_band_audio)

    # Adaptive thresholding
    def adaptive_threshold(env, percentile=95):
        return np.percentile(env, percentile)

    low_threshold = adaptive_threshold(low_env)
    mid_threshold = adaptive_threshold(mid_env)
    high_threshold = adaptive_threshold(high_env)

    # Compression
    def compress(x, threshold, ratio=4):
        mask = x > threshold
        x[mask] = threshold + (x[mask] - threshold) / ratio
        return x

    low_band_compressed = compress(low_band_audio, low_threshold)
    mid_band_compressed = compress(mid_band_audio, mid_threshold)
    high_band_compressed = compress(high_band_audio, high_threshold)

    # Mix back
    de_essed = low_band_compressed + mid_band_compressed + high_band_compressed

    return de_essed


def main() -> None:
    pyaudio_instance = pyaudio.PyAudio()
    input_stream = None
    output_stream = None

    try:
        list_audio_devices(pyaudio_instance)

        input_device_index = get_device_index(
            "Enter the index of the input device you want to use: "
        )
        input_device_info = pyaudio_instance.get_device_info_by_index(
            input_device_index
        )

        audio_format = pyaudio.paFloat32
        input_channels = input_device_info["maxInputChannels"]
        sample_rate = int(input_device_info["defaultSampleRate"])
        chunk_size = 1024
        output_channels = 2

        input_stream = pyaudio_instance.open(
            format=audio_format,
            channels=input_channels,
            rate=sample_rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=chunk_size,
        )

        output_stream = pyaudio_instance.open(
            format=audio_format,
            channels=output_channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=chunk_size,
        )

        print("Processing audio... Press Ctrl+C to stop.")
        while True:
            input_data = np.frombuffer(
                input_stream.read(chunk_size, exception_on_overflow=False),
                dtype=np.float32,
            )

            if input_channels == 1:
                input_data = np.repeat(input_data, 2)

            left_channel = input_data[::2]
            right_channel = input_data[1::2]

            # Apply de-essing
            left_de_essed = adaptive_de_esser(left_channel, sample_rate)
            right_de_essed = adaptive_de_esser(right_channel, sample_rate)

            # Apply stereo enhancement
            enhanced_left, enhanced_right = simple_stereo_enhance(
                left_de_essed, right_de_essed
            )

            enhanced_data = np.column_stack((enhanced_left, enhanced_right)).flatten()
            enhanced_data = np.clip(enhanced_data, -1, 1)

            output_stream.write(enhanced_data.astype(np.float32).tobytes())

    except KeyboardInterrupt:
        print("\nStopping audio processing.")
    except Exception as error:
        print(f"An error occurred: {error}")
    finally:
        if input_stream:
            input_stream.stop_stream()
            input_stream.close()
        if output_stream:
            output_stream.stop_stream()
            output_stream.close()
        pyaudio_instance.terminate()


if __name__ == "__main__":
    main()
