#!/usr/bin/env python3

import threading
import time
import tkinter as tk
import wave

import pyaudio
import soundfile as sf

from neural_network import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiagnosisNetwork().to(device)
weights_path = os.path.join(os.getcwd(), "model_weights.pt")
model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
model.eval()


# Function to handle audio recording
def record_audio(timer_label, done_label):
    CHANNELS = 1  # Mono audio
    SAMPLE_RATE = 44100  # Original sampling rate
    DURATION = 10  # Record duration in seconds
    FORMAT = pyaudio.paInt16  # 16-bit format
    OUTPUT_FILE = "resampled_audio.wav"  # Output file path
    TARGET_SAMPLE_RATE = 4000  # Target sampling rate
    BUFFER_SIZE = 4096

    audio = pyaudio.PyAudio()
    device_index = -1
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)["name"]

        if "CUBILUX HLMS-C5: USB Audio" in str(device_info):
            device_index = i
            break

    if device_index == -1:
        print("Couldn't find input device")

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=BUFFER_SIZE,
        input_device_index=device_index,
    )
    frames = []

    frames_len = int(SAMPLE_RATE / BUFFER_SIZE * DURATION)
    for i in range(frames_len):
        data = stream.read(BUFFER_SIZE)
        timer_label.config(text=f"Frames: {i + 1} / " + str(frames_len))
        timer_label.update_idletasks()
        frames.append(data)

    record_button.config(text="Recording complete.", state="disabled")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Convert byte data to numpy array
    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

    # Resample the audio
    record_button.config(text="Resampling...", state="disabled")
    audio_data_float = (
        audio_data.astype(np.float32) / np.iinfo(np.int16).max
    )  # Convert to float32
    resampled_audio = librosa.resample(
        audio_data_float, orig_sr=SAMPLE_RATE, target_sr=TARGET_SAMPLE_RATE
    )

    # Save to file
    print("Saving to file...")
    sf.write(OUTPUT_FILE, resampled_audio, samplerate=TARGET_SAMPLE_RATE)
    print(f"Audio saved to {OUTPUT_FILE}.")

    record_button.config(text="Computing Diagnosis from model", state="disabled")
    input_data = data_from_file(OUTPUT_FILE)
    diagnoses = get_diagnosis(device, input_data, model)

    print(diagnoses)

    # Update GUI to show "Done"
    timer_label.config(text="Time: 0s")

    if diagnoses_agree(diagnoses):
        done_label.config(text=diagnoses[0])
    else:
        done_label.config(text="Can't determine diagnosis")


def diagnoses_agree(diagnoses):
    starting = diagnoses[0]

    for diagnosis in diagnoses:
        if starting != diagnosis:
            return False

    return True


# Function to handle recording in a separate thread
def start_recording():
    record_button.config(text="Recording...", state="disabled")
    done_label.config(text="")  # Clear "Done" message
    threading.Thread(target=record_and_reset).start()


# Function to record and return to the main screen
def record_and_reset():
    record_audio(timer_label, done_label)
    time.sleep(1)
    record_button.config(text="Record", state="normal")


# Create the main application window
app = tk.Tk()
app.title("Audio Recorder")
app.geometry("800x480")

# Add a record button
record_button = tk.Button(
    app, text="Record", font=("Arial", 20), command=start_recording
)
record_button.pack(expand=True)

# Add a timer label
timer_label = tk.Label(app, text="Frames: ", font=("Arial", 16))
timer_label.pack(expand=True)

# Add a label for the "Done" message
done_label = tk.Label(app, text="", font=("Arial", 16), fg="green")
done_label.pack(expand=True)

# Run the application
app.mainloop()
