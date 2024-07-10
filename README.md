# Stereo Audio Enhancer

This Python script enhances the stereo effect of audio input in real-time. It reads audio from an input device, applies a simple stereo enhancement, and outputs the processed audio to an output device.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the script file.

2. Install the required dependencies: `pip install pyaudio numpy scipy`

Note: On some systems, you might need to install additional system-level dependencies for PyAudio:
- Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
- macOS with Homebrew: `brew install portaudio`

## Usage

1. Run the script:
python stereo_enhancer.py

2. The script will display available audio devices. Note the indices for your desired input and output devices.

3. Enter the index of the input device when prompted.

4. Enter the index of the output device when prompted.

5. The script will start processing audio. You should hear the enhanced audio from your output device.

6. To stop the script, press Ctrl+C.

## Customization

Adjust the stereo enhancement by modifying the `width` parameter in the `simple_stereo_enhance` function call. Higher values create a more pronounced stereo effect.

## Troubleshooting

- If you encounter permission issues, try running the script with elevated privileges or adjust your system's audio settings.
- Ensure chosen input and output devices are not in use by other applications.
- For audio glitches or dropouts, try increasing the `chunk_size` value in the script.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page to contribute.

## Author

Alex Moss