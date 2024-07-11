import pyaudio
import numpy as np
import argparse
import sys
import select
import threading
from processing.click_pop_removal import remove_clicks_pops
from processing.noise_reduction import noise_reduction
from processing.equalization import equalize
from processing.phase_correction import phase_correction
from processing.wow_flutter_correction import wow_flutter_correction
from processing.harmonic_excitation import harmonic_excitation
from processing.dynamic_range_expansion import dynamic_range_expansion
from processing.de_esser import sidechain_de_esser
from toggle_settings import toggle_processing_settings, print_processing_settings
from utils import list_audio_devices, simple_stereo_enhance, soft_clipper
from config import AudioProcessingConfig


def process_audio(args, config):
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

        print_processing_settings(config)
        print("Processing audio...")

        while True:
            input_data = np.frombuffer(
                input_stream.read(chunk_size, exception_on_overflow=False),
                dtype=np.float32,
            )

            if input_channels == 1:
                input_data = np.repeat(input_data, 2)

            left_channel = input_data[::2]
            right_channel = input_data[1::2]

            if config.ENABLE_PROCESSING:
                if config.ENABLE_CLICK_POP_REMOVAL:
                    left_channel = remove_clicks_pops(left_channel, sample_rate)
                    right_channel = remove_clicks_pops(right_channel, sample_rate)

                if config.ENABLE_NOISE_REDUCTION:
                    left_channel = noise_reduction(left_channel, sample_rate)
                    right_channel = noise_reduction(right_channel, sample_rate)

                if config.ENABLE_EQUALIZATION:
                    left_channel = equalize(left_channel, sample_rate)
                    right_channel = equalize(right_channel, sample_rate)

                if config.ENABLE_PHASE_CORRECTION:
                    left_channel, right_channel = phase_correction(
                        left_channel, right_channel
                    )

                if config.ENABLE_WOW_FLUTTER_CORRECTION:
                    left_channel = wow_flutter_correction(left_channel, sample_rate)
                    right_channel = wow_flutter_correction(right_channel, sample_rate)

                if config.ENABLE_HARMONIC_EXCITATION:
                    left_channel = harmonic_excitation(left_channel, sample_rate)
                    right_channel = harmonic_excitation(right_channel, sample_rate)

                if config.ENABLE_DYNAMIC_RANGE_EXPANSION:
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

                if config.ENABLE_DE_ESSING:
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

                if config.ENABLE_STEREO_ENHANCEMENT:
                    left_channel, right_channel = simple_stereo_enhance(
                        left_channel, right_channel, args.stereo_width
                    )

                if config.ENABLE_SOFT_CLIPPING:
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


def user_input_thread(config):
    while True:
        user_input = input()
        if user_input == "t":
            toggle_processing_settings(config)
            print_processing_settings(config)
            print("Processing audio...")


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

    config = AudioProcessingConfig()

    if args.disable_all:
        config.ENABLE_DE_ESSING = False
        config.ENABLE_STEREO_ENHANCEMENT = False
        config.ENABLE_SOFT_CLIPPING = False
        config.ENABLE_CLICK_POP_REMOVAL = False
        config.ENABLE_NOISE_REDUCTION = False
        config.ENABLE_EQUALIZATION = False
        config.ENABLE_PHASE_CORRECTION = False
        config.ENABLE_WOW_FLUTTER_CORRECTION = False
        config.ENABLE_HARMONIC_EXCITATION = False
        config.ENABLE_DYNAMIC_RANGE_EXPANSION = False
    else:
        config.ENABLE_DE_ESSING = not args.disable_de_essing
        config.ENABLE_STEREO_ENHANCEMENT = not args.disable_stereo
        config.ENABLE_SOFT_CLIPPING = not args.disable_clipping
        config.ENABLE_CLICK_POP_REMOVAL = not args.disable_click_pop_removal
        config.ENABLE_NOISE_REDUCTION = not args.disable_noise_reduction
        config.ENABLE_EQUALIZATION = not args.disable_equalization
        config.ENABLE_PHASE_CORRECTION = not args.disable_phase_correction
        config.ENABLE_WOW_FLUTTER_CORRECTION = not args.disable_wow_flutter_correction
        config.ENABLE_HARMONIC_EXCITATION = not args.disable_harmonic_excitation
        config.ENABLE_DYNAMIC_RANGE_EXPANSION = not args.disable_dynamic_range_expansion

    print_processing_settings(config)
    print("Processing audio...")

    audio_thread = threading.Thread(target=process_audio, args=(args, config))
    audio_thread.start()

    user_input_thread(config)


if __name__ == "__main__":
    main()
