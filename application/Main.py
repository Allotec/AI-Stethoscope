
#####################
### I M P O R T S ###
#####################

# GUI Imports
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image

# PyAudio Imports
import wave
import sys
import os
import pyaudio
import threading
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.io.wavfile import write
from scipy import signal
import librosa
import soundfile as sf
from pydub import AudioSegment



#######################################
### G L O B A L   V A R I A B L E S ###
#######################################

# Grid Size
grid_width = 10
grid_height = 10

# Textbox and Scrollbar location
tb_row_span = 3
tb_col_span = 5
tb_row = 0
tb_col = 0

# Audio Filtering
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

GUIActive = True
bell_mode = True
diaphragm_mode = False
wide_mode = False



#####################################
### A U D I O   F U N C T I O N S ###
#####################################

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        sos = butter(N=order, Wn = [lowcut, highcut], btype = 'bandpass', output = 'sos', fs = fs)
        y = sosfilt(sos, data)
        return y

def myfunction(myfile):
    with wave.open(myfile, 'rb') as wf:
        # Instantiate PyAudio and initialize PortAudio system resources (1)
        p = pyaudio.PyAudio()

        # Open stream (2)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        # Play samples from the wave file (3)
        while len(data := wf.readframes(CHUNK)):  # Requires Python 3.8+ for :=
            stream.write(data)
        # Close stream (4)
        stream.close()
        # Release PortAudio system resources (5)
        p.terminate()

def streamfunction():
    p = pyaudio.PyAudio()

    buffer = []

    # Open stream (2)
    stream = p.open(format = pyaudio.paInt16,
                    channels = 1,
                    sample_rate = 44100,
                    input = True,
                    frames_per_buffer = 1024)
    
    buffer = stream.read(1024)
    
    stream.close()

    # Release PortAudio system resources (5)
    p.terminate()

def playback_example():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "voice.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    myfunction(WAVE_OUTPUT_FILENAME)

def stream_example():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "voice.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output = True,
                        frames_per_buffer=CHUNK)

    while True:
        data = stream.read(CHUNK)
        stream.write(data)

def filtrationstation():

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output = True,
                        frames_per_buffer=CHUNK)
    
    while GUIActive:

        while bell_mode:        #  20 -  200 Hz
            data = stream.read(CHUNK)
            #filtereddata = butter_bandpass_filter(data, 20, 200, RATE, 5)
            #stream.write(filtereddata)
            stream.write(data)

        while diaphragm_mode:   # 200 - 2000 Hz
            data = stream.read(CHUNK)
            #filtereddata = butter_bandpass_filter(data, 200, 2000, RATE, 5)
            #stream.write(filtereddata)
            stream.write(data)

        while wide_mode:        #  20 - 2000 Hz
            data = stream.read(CHUNK)
            #filtereddata = butter_bandpass_filter(data, 20, 2000, RATE, 5)
            #stream.write(filtereddata)
            stream.write(data)



#######################################
### B U T T O N   F U N C T I O N S ###
#######################################

# Button 1 Bell Mode
# Prints text to textbox when pressed
def but_bell():
    global bell_mode, diaphragm_mode, wide_mode
    diaphragm_mode = False 
    wide_mode = False
    bell_mode = True
    insert_text("Bell Mode")
    
# Button 2 Diaphragm Mode
# Prints text to textbox when pressed
def but_diaphragm():
    global bell_mode, diaphragm_mode, wide_mode
    bell_mode = False 
    wide_mode = False
    diaphragm_mode = True
    insert_text("Diaphragm Mode")
    
# Button 3 Wide Mode
# Prints text to textbox when pressed
def but_wide():
    global bell_mode, diaphragm_mode, wide_mode
    bell_mode = False
    diaphragm_mode = False
    wide_mode = True
    insert_text("Wide Mode")

# Insert Text
# Prints display_text text to textbox and adjusts scrollbar position to bottom 
def insert_text(display_text):
    text.insert(END, display_text + "\n") # inserts text at end of textbox
    text.yview_moveto(1.0) # sets scrollbar and textbox position to bottom
    print(display_text)
    


#######################################################
### G R A P H I C A L   U S E R   I N T E R F A C E ###
#######################################################

# Main Window
root = Tk()
root.geometry('600x475')
root.title("Remote Stethoscope")
frm = ttk.Frame(root, padding=10)
frm.grid()
style_color = ttk.Style()
style_color.configure("color.TButton", background="white")

# Bell Mode Button
but_bell_mode = ttk.Button(frm, text="Bell", command=but_bell, style="color.TButton")
but_bell_mode.grid(column=0, row=tb_row_span, sticky="nesw")
img_low_freq = ImageTk.PhotoImage(Image.open(os.getcwd() + "\low_frequency.png").resize((100,25)))
img_bell_mode = ttk.Label(frm, image=img_low_freq)
img_bell_mode.grid(column=0, row=tb_row_span+1)
img_bell_mode.configure(background='white')

# Diaphragm Mode Button
but_diaphragm_mode = ttk.Button(frm, text="Diaphragm", command=but_diaphragm)
but_diaphragm_mode.grid(column=1, row=tb_row_span, sticky="nesw")
img_high_freq = ImageTk.PhotoImage(Image.open(os.getcwd() + "\high_frequency.png").resize((100,25)))
img_diaphragm_mode = ttk.Label(frm, image=img_high_freq)
img_diaphragm_mode.grid(column=1, row=tb_row_span+1)

# Wide Mode Button
but_wide_mode = ttk.Button(frm, text="Wide", command=but_wide)
but_wide_mode.grid(column=2, row=tb_row_span, sticky="nesw")
img_range_freq = ImageTk.PhotoImage(Image.open(os.getcwd() + "\\range_frequency.png").resize((100,25)))
img_wide_mode = ttk.Label(frm, image=img_range_freq)
img_wide_mode.grid(column=2, row=tb_row_span+1)

# Scrolling Text Box
scroll = ttk.Scrollbar(frm, orient='vertical')
scroll.grid(column=(tb_col+tb_col_span), row=tb_row, columnspan=1, rowspan=tb_row_span, sticky='ns')
text = Text(frm, font=("Georgia, 10"), yscrollcommand=scroll.set)
text.grid(column=tb_col, row=tb_row, columnspan=tb_col_span, rowspan=tb_row_span)
scroll.config(command=text.yview)



#################################
### P R O G R A M   S T A R T ###
#################################

threading.Thread(target=filtrationstation).start()
root.mainloop()
GUIActive = False
bell_mode = False
diaphragm_mode = False
wide_mode = False