import os
import sys
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, filename):
        self.filename = filename

    def on_modified(self, event):
        if event.src_path == self.filename:
            print(f"Code change detected in {self.filename}")
            restart_audio_processing()

def restart_audio_processing():
    global audio_process
    if audio_process:
        audio_process.terminate()
        audio_process.wait()
    audio_process = subprocess.Popen(
        [sys.executable, "audio_processor.py"] + sys.argv[1:]
    )

if __name__ == "__main__":
    audio_process = None

    if "--list-devices" in sys.argv:
        subprocess.run([sys.executable, "audio_processor.py", "--list-devices"])
        sys.exit(0)

    if "--input-device" not in sys.argv:
        subprocess.run([sys.executable, "audio_processor.py", "--list-devices"])
        input_device = input("Enter the index of the input device you want to use: ")
        sys.argv.extend(["--input-device", input_device])

    restart_audio_processing()

    event_handler = CodeChangeHandler(os.path.abspath("audio_processor.py"))
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    if audio_process:
        audio_process.terminate()
        audio_process.wait()
