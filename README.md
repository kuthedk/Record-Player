# Real-time Stereo Audio Enhancer with Live Updating

This Python project enhances stereo audio in real-time, featuring de-essing, stereo enhancement, and soft clipping. It reads audio from an input device, applies processing, and outputs the processed audio to an output device. The script can be modified while running, with changes taking effect immediately.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the script files.

2. Install the required dependencies:
`pip install pyaudio numpy scipy watchdog pynput`

Note: On some systems, you might need to install additional system-level dependencies for PyAudio:
- Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
- macOS with Homebrew: `brew install portaudio`

## Usage

1. To list available audio devices:
`python main.py --list-devices`

2. To start the audio processing:
`python main.py [options]`

If you don't specify an input device, the script will list available devices and prompt you to choose one.

Available options:
- `--input-device`: Input device index
- `--output-device`: Output device index (default: system default)
- `--chunk-size`: Audio chunk size (default: 2048)
- `--stereo-width`: Stereo enhancement width (default: 1.5)
- `--clip-threshold`: Soft clipper threshold (default: 0.95)
- `--output-gain`: Output gain (default: 0.9)
- `--de-essing-strength`: De-essing strength (default: 1.0)
- `--disable-de-essing`: Disable de-essing
- `--disable-stereo`: Disable stereo enhancement
- `--disable-clipping`: Disable soft clipping
- `--disable-all`: Disable all processing (de-essing, stereo enhancement, and soft clipping)

3. Once the script is running:
- Press the spacebar to toggle all audio processing on/off.
- Edit the `audio_processor.py` file to modify the audio processing code. Changes will take effect immediately upon saving the file.

4. To stop the script, press Ctrl+C.

Examples:
`python main.py --stereo-width 1.8 --clip-threshold 0.98 --output-gain 0.85 --de-essing-strength 1.2 --disable-clipping`
This command will list available devices, prompt you to choose an input device, then start processing with the specified parameters and soft clipping disabled.
`python main.py --disable-all`
This command will start the script with all processing (de-essing, stereo enhancement, and soft clipping) disabled, effectively passing the audio through without modification.

## Customization

You can customize the audio processing by modifying the `audio_processor.py` file. The script uses hot-reloading, so any changes you make to the file will take effect immediately without needing to restart the script.

Areas you might want to customize include:
- Adjusting the frequency bands in the `adaptive_de_esser` function
- Modifying the compression ratios in the de-esser
- Changing the stereo enhancement algorithm in `simple_stereo_enhance`
- Adjusting the soft clipper function

## Features

1. **De-essing**: Reduces sibilance in the audio signal using a multi-band approach.
2. **Stereo Enhancement**: Widens the stereo image for a more immersive sound.
3. **Soft Clipping**: Prevents harsh digital clipping by smoothly limiting the signal.
4. **Live Updates**: Modify the processing code while the script is running.
5. **Togglable Processing**: Use the spacebar to enable/disable all processing in real-time.
6. **Customizable Parameters**: Adjust various parameters through command-line arguments.

## Troubleshooting

- If you encounter permission issues, try running the script with elevated privileges or adjust your system's audio settings.
- Ensure the chosen input device is not in use by other applications.
- For audio glitches or dropouts, try increasing the `chunk_size` value in the script.
- If changes don't seem to take effect, check the console for any error messages that might indicate syntax errors in your modifications.

## Known Limitations

- The script currently supports up to 2-channel (stereo) audio. Multichannel audio beyond stereo is not supported.
- Very low latency applications might experience some delay due to the processing overhead.
- The de-essing algorithm may need fine-tuning for different voice types or audio sources.

## Future Improvements

- Add support for multichannel audio beyond stereo
- Implement more advanced stereo enhancement techniques
- Add a graphical user interface for easier parameter adjustment
- Incorporate more audio effects like EQ, compression, and reverb

## License

This project is open source and available under the MIT License.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page or submit pull requests to improve the project.

## Acknowledgements

- The PyAudio team for providing the audio I/O functionality
- The NumPy and SciPy communities for their excellent scientific computing libraries
- The Watchdog project for enabling the live update feature

## Contact

If you have any questions, feel free to reach out to the project maintainer or open an issue on the project's GitHub page.

## Author

Alex Moss