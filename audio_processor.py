import pyaudio
import numpy as np
import argparse
from pynput import keyboard
from processing.click_pop_removal import remove_clicks_pops
from processing.noise_reduction import noise_reduction
from processing.equalization import equalize
from processing.phase_correction import phase_correction
from processing.wow_flutter_correction import wow_flutter_correction
from processing.harmonic_excitation import harmonic_excitation
from processing.dynamic_range_expansion import dynamic_range_expansion
from processing.de_esser import sidechain_de_esser

# Global variables for processing control
ENABLE_PROCESSING = True
ENABLE_DE_ESSING = True
ENABLE_STEREO_ENHANCEMENT = True
ENABLE_SOFT_CLIPPING = True
ENABLE_CLICK_POP_REMOVAL = True
ENABLE_NOISE_REDUCTION = True
ENABLE_EQUALIZATION = True
ENABLE_PHASE_CORRECTION = True
ENABLE_WOW_FLUTTER_CORRECTION = True
ENABLE_HARMONIC_EXCITATION = True
ENABLE_DYNAMIC_RANGE_EXPANSION = True


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


def on_press(key):
    global ENABLE_PROCESSING
    if key == keyboard.Key.space:
        ENABLE_PROCESSING = not ENABLE_PROCESSING
        print(f"Processing {'enabled' if ENABLE_PROCESSING else 'disabled'}")


def process_audio(args):
    global ENABLE_PROCESSING, ENABLE_DE_ESSING, ENABLE_STEREO_ENHANCEMENT, ENABLE_SOFT_CLIPPING
    global ENABLE_CLICK_POP_REMOVAL, ENABLE_NOISE_REDUCTION, ENABLE_EQUALIZATION
    global ENABLE_PHASE_CORRECTION, ENABLE_WOW_FLUTTER_CORRECTION, ENABLE_HARMONIC_EXCITATION
    global ENABLE_DYNAMIC_RANGE_EXPANSION

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
                if ENABLE_CLICK_POP_REMOVAL:
                    left_channel = remove_clicks_pops(left_channel, sample_rate)
                    right_channel = remove_clicks_pops(right_channel, sample_rate)

                if ENABLE_NOISE_REDUCTION:
                    left_channel = noise_reduction(left_channel, sample_rate)
                    right_channel = noise_reduction(right_channel, sample_rate)

                if ENABLE_EQUALIZATION:
                    left_channel = equalize(left_channel, sample_rate)
                    right_channel = equalize(right_channel, sample_rate)

                if ENABLE_PHASE_CORRECTION:
                    left_channel, right_channel = phase_correction(
                        left_channel, right_channel
                    )

                if ENABLE_WOW_FLUTTER_CORRECTION:
                    left_channel = wow_flutter_correction(left_channel, sample_rate)
                    right_channel = wow_flutter_correction(right_channel, sample_rate)

                if ENABLE_HARMONIC_EXCITATION:
                    left_channel = harmonic_excitation(left_channel, sample_rate)
                    right_channel = harmonic_excitation(right_channel, sample_rate)

                if ENABLE_DYNAMIC_RANGE_EXPANSION:
                    left_channel = dynamic_range_expansion(
                        left_channel,
                        sample_rate,
                        args.dre_threshold,
                        args.dre_ratio,
                        args.dre_attack,
                        args.dre_release,
                    )
                    right_channel = dynamic_range_expansion(
                        right_channel,
                        sample_rate,
                        args.dre_threshold,
                        args.dre_ratio,
                        args.dre_attack,
                        args.dre_release,
                    )

                if ENABLE_DE_ESSING:
                    left_channel = sidechain_de_esser(
                        left_channel,
                        sample_rate,
                        args.de_essing_threshold,
                        args.de_essing_ratio,
                        args.de_essing_attack,
                        args.de_essing_release,
                    )
                    right_channel = sidechain_de_esser(
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
        "--dre-threshold",
        type=float,
        default=0.2,
        help="Dynamic Range Expansion threshold (default: 0.2)",
    )
    parser.add_argument(
        "--dre-ratio",
        type=float,
        default=1.5,
        help="Dynamic Range Expansion ratio (default: 1.5)",
    )
    parser.add_argument(
        "--dre-attack",
        type=float,
        default=0.01,
        help="Dynamic Range Expansion attack time in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--dre-release",
        type=float,
        default=0.1,
        help="Dynamic Range Expansion release time in seconds (default: 0.1)",
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
        "--disable-click-pop-removal",
        action="store_true",
        help="Disable click and pop removal",
    )
    parser.add_argument(
        "--disable-noise-reduction", action="store_true", help="Disable noise reduction"
    )
    parser.add_argument(
        "--disable-equalization", action="store_true", help="Disable equalization"
    )
    parser.add_argument(
        "--disable-phase-correction",
        action="store_true",
        help="Disable phase correction",
    )
    parser.add_argument(
        "--disable-wow-flutter-correction",
        action="store_true",
        help="Disable wow and flutter correction",
    )
    parser.add_argument(
        "--disable-harmonic-excitation",
        action="store_true",
        help="Disable harmonic excitation",
    )
    parser.add_argument(
        "--disable-dynamic-range-expansion",
        action="store_true",
        help="Disable dynamic range expansion",
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
    global ENABLE_CLICK_POP_REMOVAL, ENABLE_NOISE_REDUCTION, ENABLE_EQUALIZATION
    global ENABLE_PHASE_CORRECTION, ENABLE_WOW_FLUTTER_CORRECTION, ENABLE_HARMONIC_EXCITATION
    global ENABLE_DYNAMIC_RANGE_EXPANSION
    if args.disable_all:
        ENABLE_DE_ESSING = False
        ENABLE_STEREO_ENHANCEMENT = False
        ENABLE_SOFT_CLIPPING = False
        ENABLE_CLICK_POP_REMOVAL = False
        ENABLE_NOISE_REDUCTION = False
        ENABLE_EQUALIZATION = False
        ENABLE_PHASE_CORRECTION = False
        ENABLE_WOW_FLUTTER_CORRECTION = False
        ENABLE_HARMONIC_EXCITATION = False
        ENABLE_DYNAMIC_RANGE_EXPANSION = False
    else:
        ENABLE_DE_ESSING = not args.disable_de_essing
        ENABLE_STEREO_ENHANCEMENT = not args.disable_stereo
        ENABLE_SOFT_CLIPPING = not args.disable_clipping
        ENABLE_CLICK_POP_REMOVAL = not args.disable_click_pop_removal
        ENABLE_NOISE_REDUCTION = not args.disable_noise_reduction
        ENABLE_EQUALIZATION = not args.disable_equalization
        ENABLE_PHASE_CORRECTION = not args.disable_phase_correction
        ENABLE_WOW_FLUTTER_CORRECTION = not args.disable_wow_flutter_correction
        ENABLE_HARMONIC_EXCITATION = not args.disable_harmonic_excitation
        ENABLE_DYNAMIC_RANGE_EXPANSION = not args.disable_dynamic_range_expansion

    print("Current processing settings:")
    print(f"  De-essing: {'Enabled' if ENABLE_DE_ESSING else 'Disabled'}")
    print(
        f"  Stereo Enhancement: {'Enabled' if ENABLE_STEREO_ENHANCEMENT else 'Disabled'}"
    )
    print(f"  Soft Clipping: {'Enabled' if ENABLE_SOFT_CLIPPING else 'Disabled'}")
    print(
        f"  Click and Pop Removal: {'Enabled' if ENABLE_CLICK_POP_REMOVAL else 'Disabled'}"
    )
    print(f"  Noise Reduction: {'Enabled' if ENABLE_NOISE_REDUCTION else 'Disabled'}")
    print(f"  Equalization: {'Enabled' if ENABLE_EQUALIZATION else 'Disabled'}")
    print(f"  Phase Correction: {'Enabled' if ENABLE_PHASE_CORRECTION else 'Disabled'}")
    print(
        f"  Wow and Flutter Correction: {'Enabled' if ENABLE_WOW_FLUTTER_CORRECTION else 'Disabled'}"
    )
    print(
        f"  Harmonic Excitation: {'Enabled' if ENABLE_HARMONIC_EXCITATION else 'Disabled'}"
    )
    print(
        f"  Dynamic Range Expansion: {'Enabled' if ENABLE_DYNAMIC_RANGE_EXPANSION else 'Disabled'}"
    )

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    process_audio(args)


if __name__ == "__main__":
    main()
