#!/usr/bin/env python3

import threading
import time
import tkinter as tk
import wave

import pyaudio

from neural_network import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiagnosisNetwork().to(device)
weights_path = os.path.join(os.getcwd(), "model_weights.pt")
model.load_state_dict(torch.load(weights_path, weights_only=True))
model.eval()


# Function to handle audio recording
def record_audio(timer_label, done_label):
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "output.wav"
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 4000
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    frames = []
    for i in range(RECORD_SECONDS):
        time.sleep(1)
        timer_label.config(text=f"Time: {i + 1}s")
        timer_label.update_idletasks()
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    input_data = data_from_file(WAVE_OUTPUT_FILENAME)
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
timer_label = tk.Label(app, text="Time: 0s", font=("Arial", 16))
timer_label.pack(expand=True)

# Add a label for the "Done" message
done_label = tk.Label(app, text="", font=("Arial", 16), fg="green")
done_label.pack(expand=True)

# Run the application
app.mainloop()
