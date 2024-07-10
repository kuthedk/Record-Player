import pyaudio
import numpy as np
from scipy import signal


def list_audio_devices(pyaudio_instance: pyaudio.PyAudio) -> None:
    """Print all available audio devices."""
    print("Available audio devices:")
    for device_index in range(pyaudio_instance.get_device_count()):
        device_info = pyaudio_instance.get_device_info_by_index(device_index)
        print(f"Device {device_index}: {device_info['name']}")
        print(f"  Max Input Channels: {device_info['maxInputChannels']}")
        print(f"  Max Output Channels: {device_info['maxOutputChannels']}")
        print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
        print()


def get_device_index(prompt: str) -> int:
    """Get user input for device selection."""
    return int(input(prompt))


def simple_stereo_enhance(
    left_channel: np.ndarray, right_channel: np.ndarray, width: float = 1.8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply simple stereo enhancement to audio channels.

    Args:
    left_channel (np.ndarray): Left channel audio data.
    right_channel (np.ndarray): Right channel audio data.
    width (float): Stereo width factor. Default is 1.8.

    Returns:
    tuple[np.ndarray, np.ndarray]: Enhanced left and right channel audio data.
    """
    mid_channel = (left_channel + right_channel) / 2
    side_channel = (left_channel - right_channel) / 2
    enhanced_left = mid_channel + width * side_channel
    enhanced_right = mid_channel - width * side_channel
    return enhanced_left, enhanced_right


def main() -> None:
    # Initialize PyAudio
    pyaudio_instance = pyaudio.PyAudio()

    # List available audio devices
    list_audio_devices(pyaudio_instance)

    # Get user input for device selection
    input_device_index = get_device_index(
        "Enter the index of the input device you want to use: "
    )
    output_device_index = get_device_index(
        "Enter the index of the output device you want to use: "
    )

    # Get device info
    input_device_info = pyaudio_instance.get_device_info_by_index(input_device_index)
    output_device_info = pyaudio_instance.get_device_info_by_index(output_device_index)

    # Audio parameters
    audio_format = pyaudio.paFloat32
    input_channels = min(2, input_device_info["maxInputChannels"])
    sample_rate = int(input_device_info["defaultSampleRate"])
    chunk_size = 1024
    output_channels = 2

    print(
        f"Using {input_channels} input channel(s) and {output_channels} output channel(s)"
    )

    # Open input stream
    input_stream = pyaudio_instance.open(
        format=audio_format,
        channels=input_channels,
        rate=sample_rate,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=chunk_size,
    )

    # Open output stream
    output_stream = pyaudio_instance.open(
        format=audio_format,
        channels=output_channels,
        rate=sample_rate,
        output=True,
        output_device_index=output_device_index,
        frames_per_buffer=chunk_size,
    )

    print("Recording and processing. Press Ctrl+C to stop.")
    try:
        while True:
            # Read input
            input_data = np.frombuffer(input_stream.read(chunk_size), dtype=np.float32)

            # If input is mono, duplicate the channel for stereo processing
            if input_channels == 1:
                input_data = np.repeat(input_data, 2)

            # Split channels
            left_channel = input_data[::2]
            right_channel = input_data[1::2]

            # Apply simple stereo enhancement
            enhanced_left, enhanced_right = simple_stereo_enhance(
                left_channel, right_channel
            )

            # Combine channels
            enhanced_data = np.column_stack((enhanced_left, enhanced_right)).flatten()

            # Clip to prevent overflow
            enhanced_data = np.clip(enhanced_data, -1, 1)

            # Write to output stream
            output_stream.write(enhanced_data.astype(np.float32).tobytes())

    except KeyboardInterrupt:
        print("Stopped recording")
    except Exception as error:
        print(f"An error occurred: {error}")
    finally:
        # Clean up
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        pyaudio_instance.terminate()


if __name__ == "__main__":
    main()
