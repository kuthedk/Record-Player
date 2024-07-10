import pyaudio
import numpy as np
from scipy import signal
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def list_audio_devices(pyaudio_instance: pyaudio.PyAudio) -> None:
    """Print all available audio devices."""
    logging.info("Listing available audio devices:")
    for device_index in range(pyaudio_instance.get_device_count()):
        device_info = pyaudio_instance.get_device_info_by_index(device_index)
        logging.info(f"Device {device_index}: {device_info['name']}")
        logging.info(f"  Max Input Channels: {device_info['maxInputChannels']}")
        logging.info(f"  Max Output Channels: {device_info['maxOutputChannels']}")
        logging.info(f"  Default Sample Rate: {device_info['defaultSampleRate']}")


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
    pyaudio_instance = pyaudio.PyAudio()
    input_stream = None
    output_stream = None

    try:
        list_audio_devices(pyaudio_instance)

        input_device_index = get_device_index(
            "Enter the index of the input device you want to use: "
        )
        logging.info(f"Selected input device index: {input_device_index}")

        input_device_info = pyaudio_instance.get_device_info_by_index(
            input_device_index
        )
        logging.info(f"Input device info: {input_device_info}")

        audio_format = pyaudio.paFloat32
        input_channels = min(2, input_device_info["maxInputChannels"])
        sample_rate = int(input_device_info["defaultSampleRate"])
        chunk_size = 1024
        output_channels = 2

        logging.info(
            f"Using {input_channels} input channel(s) and {output_channels} output channel(s)"
        )
        logging.info(f"Sample rate: {sample_rate}, Chunk size: {chunk_size}")

        logging.info("Opening input stream...")
        input_stream = pyaudio_instance.open(
            format=audio_format,
            channels=input_channels,
            rate=sample_rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=chunk_size,
        )
        logging.info("Input stream opened successfully")

        logging.info("Opening output stream...")
        output_stream = pyaudio_instance.open(
            format=audio_format,
            channels=output_channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=chunk_size,
        )
        logging.info("Output stream opened successfully")

        logging.info("Starting audio processing loop")
        while True:
            try:
                input_data = np.frombuffer(
                    input_stream.read(chunk_size, exception_on_overflow=False),
                    dtype=np.float32,
                )

                if input_channels == 1:
                    input_data = np.repeat(input_data, 2)

                left_channel = input_data[::2]
                right_channel = input_data[1::2]

                enhanced_left, enhanced_right = simple_stereo_enhance(
                    left_channel, right_channel
                )

                enhanced_data = np.column_stack(
                    (enhanced_left, enhanced_right)
                ).flatten()
                enhanced_data = np.clip(enhanced_data, -1, 1)

                output_stream.write(enhanced_data.astype(np.float32).tobytes())
            except IOError as e:
                logging.error(f"IOError occurred: {e}")
                if e.errno == -9981:  # Input overflow
                    logging.warning("Input overflow detected. Continuing...")
                    continue
                else:
                    raise

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping recording.")
    except Exception as error:
        logging.exception(f"An unexpected error occurred: {error}")
    finally:
        logging.info("Cleaning up resources...")
        if input_stream is not None:
            try:
                logging.info("Stopping input stream...")
                input_stream.stop_stream()
                logging.info("Closing input stream...")
                input_stream.close()
            except Exception as e:
                logging.error(f"Error while closing input stream: {e}")

        if output_stream is not None:
            try:
                logging.info("Stopping output stream...")
                output_stream.stop_stream()
                logging.info("Closing output stream...")
                output_stream.close()
            except Exception as e:
                logging.error(f"Error while closing output stream: {e}")

        try:
            logging.info("Terminating PyAudio...")
            pyaudio_instance.terminate()
        except Exception as e:
            logging.error(f"Error while terminating PyAudio: {e}")

        logging.info("Cleanup completed.")


if __name__ == "__main__":
    main()
