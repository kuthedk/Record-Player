import pyaudio
import numpy as np


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
