import pyaudio
import numpy as np
from scipy import signal
import argparse
from pynput import keyboard

# Global variables for processing control
ENABLE_PROCESSING = True
ENABLE_DE_ESSING = True
ENABLE_STEREO_ENHANCEMENT = True
ENABLE_SOFT_CLIPPING = True


def list_audio_devices():
    p = pyaudio.PyAudio()
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        print(f"Device {i}: {dev['name']}")
        print(f"  Max Input Channels: {dev['maxInputChannels']}")
        print(f"  Max Output Channels: {dev['maxOutputChannels']}")
        print(f"  Default Sample Rate: {dev['defaultSampleRate']}")
    p.terminate()


def simple_stereo_enhance(
    left_channel: np.ndarray, right_channel: np.ndarray, width: float
) -> tuple[np.ndarray, np.ndarray]:
    mid_channel = (left_channel + right_channel) / 2
    side_channel = (left_channel - right_channel) / 2
    enhanced_left = mid_channel + width * side_channel
    enhanced_right = mid_channel - width * side_channel
    return enhanced_left, enhanced_right


def soft_clipper(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.tanh(x / threshold) * threshold


def de_esser(audio, sample_rate, threshold, ratio, attack, release):
    sibilance_freq_range = [5000, 10000]
    nyquist = 0.5 * sample_rate
    low = sibilance_freq_range[0] / nyquist
    high = sibilance_freq_range[1] / nyquist
    b, a = signal.butter(4, [low, high], btype="band")

    sibilance = signal.lfilter(b, a, audio)

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

        gain_reduction[i] = np.maximum(0, 1 - threshold / (envelope[i] + 1e-6))
        gain_reduction[i] = np.minimum(gain_reduction[i], 1.0 / ratio)

    compressed_sibilance = sibilance * (1 - gain_reduction)
    de_essed = audio - compressed_sibilance
    return de_essed


def on_press(key):
    global ENABLE_PROCESSING
    if key == keyboard.Key.space:
        ENABLE_PROCESSING = not ENABLE_PROCESSING
        print(f"Processing {'enabled' if ENABLE_PROCESSING else 'disabled'}")


def process_audio(args):
    global ENABLE_PROCESSING, ENABLE_DE_ESSING, ENABLE_STEREO_ENHANCEMENT, ENABLE_SOFT_CLIPPING

    pyaudio_instance = pyaudio.PyAudio()
    input_stream = None
    output_stream = None

    try:
        input_device_info = pyaudio_instance.get_device_info_by_index(args.input_device)

        audio_format = pyaudio.paFloat32
        input_channels = min(
            2, input_device_info["maxInputChannels"]
        )  # Ensure max 2 channels
        sample_rate = int(input_device_info["defaultSampleRate"])
        chunk_size = args.chunk_size
        output_channels = 2

        input_stream = pyaudio_instance.open(
            format=audio_format,
            channels=input_channels,
            rate=sample_rate,
            input=True,
            input_device_index=args.input_device,
            frames_per_buffer=chunk_size,
        )

        output_stream = pyaudio_instance.open(
            format=audio_format,
            channels=output_channels,
            rate=sample_rate,
            output=True,
            output_device_index=args.output_device,
            frames_per_buffer=chunk_size,
        )

        print("Processing audio... Press space to toggle processing on/off.")
        while True:
            input_data = np.frombuffer(
                input_stream.read(chunk_size, exception_on_overflow=False),
                dtype=np.float32,
            )

            if input_channels == 1:
                input_data = np.repeat(input_data, 2)

            left_channel = input_data[::2]
            right_channel = input_data[1::2]

            if ENABLE_PROCESSING:
                if ENABLE_DE_ESSING:
                    left_channel = de_esser(
                        left_channel,
                        sample_rate,
                        args.de_essing_threshold,
                        args.de_essing_ratio,
                        args.de_essing_attack,
                        args.de_essing_release,
                    )
                    right_channel = de_esser(
                        right_channel,
                        sample_rate,
                        args.de_essing_threshold,
                        args.de_essing_ratio,
                        args.de_essing_attack,
                        args.de_essing_release,
                    )

                if ENABLE_STEREO_ENHANCEMENT:
                    left_channel, right_channel = simple_stereo_enhance(
                        left_channel, right_channel, args.stereo_width
                    )

                if ENABLE_SOFT_CLIPPING:
                    left_channel = soft_clipper(left_channel, args.clip_threshold)
                    right_channel = soft_clipper(right_channel, args.clip_threshold)

            enhanced_data = np.column_stack((left_channel, right_channel)).flatten()

            # Final gain adjustment
            enhanced_data *= args.output_gain

            output_stream.write(enhanced_data.astype(np.float32).tobytes())

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


def main():
    parser = argparse.ArgumentParser(description="Real-time Stereo Audio Enhancer")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument("--input-device", type=int, help="Input device index")
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Output device index (default: system default)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2048, help="Audio chunk size (default: 2048)"
    )
    parser.add_argument(
        "--stereo-width",
        type=float,
        default=1.5,
        help="Stereo enhancement width (default: 1.5)",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.95,
        help="Soft clipper threshold (default: 0.95)",
    )
    parser.add_argument(
        "--output-gain", type=float, default=0.9, help="Output gain (default: 0.9)"
    )
    parser.add_argument(
        "--de-essing-threshold",
        type=float,
        default=0.2,
        help="De-essing threshold (default: 0.2)",
    )
    parser.add_argument(
        "--de-essing-ratio",
        type=float,
        default=4.0,
        help="De-essing ratio (default: 4.0)",
    )
    parser.add_argument(
        "--de-essing-attack",
        type=float,
        default=0.001,
        help="De-essing attack time in seconds (default: 0.001)",
    )
    parser.add_argument(
        "--de-essing-release",
        type=float,
        default=0.05,
        help="De-essing release time in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--disable-de-essing", action="store_true", help="Disable de-essing"
    )
    parser.add_argument(
        "--disable-stereo", action="store_true", help="Disable stereo enhancement"
    )
    parser.add_argument(
        "--disable-clipping", action="store_true", help="Disable soft clipping"
    )
    parser.add_argument(
        "--disable-all",
        action="store_true",
        help="Disable all processing (de-essing, stereo enhancement, and soft clipping)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.input_device is None:
        list_audio_devices()
        args.input_device = int(
            input("Enter the index of the input device you want to use: ")
        )

    global ENABLE_DE_ESSING, ENABLE_STEREO_ENHANCEMENT, ENABLE_SOFT_CLIPPING
    if args.disable_all:
        ENABLE_DE_ESSING = False
        ENABLE_STEREO_ENHANCEMENT = False
        ENABLE_SOFT_CLIPPING = False
    else:
        ENABLE_DE_ESSING = not args.disable_de_essing
        ENABLE_STEREO_ENHANCEMENT = not args.disable_stereo
        ENABLE_SOFT_CLIPPING = not args.disable_clipping

    print("Current processing settings:")
    print(f"  De-essing: {'Enabled' if ENABLE_DE_ESSING else 'Disabled'}")
    print(
        f"  Stereo Enhancement: {'Enabled' if ENABLE_STEREO_ENHANCEMENT else 'Disabled'}"
    )
    print(f"  Soft Clipping: {'Enabled' if ENABLE_SOFT_CLIPPING else 'Disabled'}")

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    process_audio(args)


if __name__ == "__main__":
    main()
