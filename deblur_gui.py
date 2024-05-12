import tkinter as tk
# from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog, ttk
import threading
import torch
from basicsr.models import create_model
from basicsr.utils.options import parse
from deblur_functions import process_video, create_model

DEFAULT_BATCH_SIZE = 32

def select_input_file():
    file_path.set(filedialog.askopenfilename(title="Select a file", filetypes=[("Video files", "*.mp4;*.avi"), ("Image files", "*.jpg;*.png")]))
    check_button_state()

def select_output_folder():
    folder_path.set(filedialog.askdirectory(title="Select output folder"))
    check_button_state()

def check_button_state():
    if file_path.get() and folder_path.get():
        start_button.config(state='normal')
    else:
        start_button.config(state='disabled')

def start_processing():
    start_button.config(text="Stop", command=stop_processing)
    threading.Thread(target=run_process, daemon=True).start()

def stop_processing():
    stop_event.set()
    start_button.config(text="Start", command=start_processing)
    progress_bar['value'] = 0
    check_button_state()

def update_progress(processed, total):
    progress = int((processed / total) * 100)
    progress_bar['value'] = progress
    app.update_idletasks()  # Update GUI elements

def run_process():
    stop_event.clear()
    batch_size = int(batch_size_entry.get()) if batch_size_entry.get().isdigit() else DEFAULT_BATCH_SIZE
    
    opt_path = './models/NAFNet-width64.yml'
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    model = create_model(opt)
    process_video(model, file_path.get(), folder_path.get(), batch_size, update_progress)
    stop_processing()

def drop(event):
    file_path.set(event.data)

app = tk.Tk()
app.title("File Processor")

file_path = tk.StringVar()
folder_path = tk.StringVar()
batch_size_entry = tk.StringVar(value="32")
stop_event = threading.Event()

# Layout
input_frame = ttk.Frame(app)
input_frame.pack(pady=20, padx=20, fill='x')

file_label = ttk.Label(input_frame, text="Input File:")
file_label.pack(side='left', padx=(0, 10))

file_entry = ttk.Entry(input_frame, textvariable=file_path, width=50)
file_entry.pack(side='left', fill='x', expand=True)
# file_entry.drop_target_register(DND_FILES)
# file_entry.dnd_bind('<<Drop>>', drop)

file_button = ttk.Button(input_frame, text="Browse...", command=select_input_file)
file_button.pack(side='left', padx=(10, 0))

output_frame = ttk.Frame(app)
output_frame.pack(pady=10, padx=20, fill='x')

folder_label = ttk.Label(output_frame, text="Output Folder:")
folder_label.pack(side='left', padx=(0, 10))

folder_entry = ttk.Entry(output_frame, textvariable=folder_path, width=50)
folder_entry.pack(side='left', fill='x', expand=True)

folder_button = ttk.Button(output_frame, text="Browse...", command=select_output_folder)
folder_button.pack(side='left', padx=(10, 0))

batch_size_frame = ttk.Frame(app)
batch_size_frame.pack(pady=10, padx=20, fill='x')

batch_size_label = ttk.Label(batch_size_frame, text="Batch Size:")
batch_size_label.pack(side='left', padx=(0, 10))

batch_size_entry = ttk.Entry(batch_size_frame, textvariable=batch_size_entry, width=10)
batch_size_entry.pack(side='left')

start_button = ttk.Button(app, text="Start", command=start_processing, state='disabled')
start_button.pack(pady=(10, 20))

progress_bar = ttk.Progressbar(app, orient='horizontal', length=200, mode='determinate', maximum=100)
progress_bar.pack(pady=(0, 20))

app.mainloop()