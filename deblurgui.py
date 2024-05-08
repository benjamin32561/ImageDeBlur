import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog, messagebox, ttk
import threading
import time

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

def run_process():
    stop_event.clear()


    total_time = 10  # Total time in seconds
    increment = 100 / total_time  # Increment per second

    for i in range(total_time):
        if stop_event.is_set():
            return
        time.sleep(1)  # Simulate processing delay
        progress_bar['value'] += increment  # Update progress bar

    stop_processing()  # Reset after completion

def drop(event):
    file_path.set(event.data)

app = TkinterDnD.Tk()
app.title("File Processor")

file_path = tk.StringVar()
folder_path = tk.StringVar()
stop_event = threading.Event()

# Layout
input_frame = ttk.Frame(app)
input_frame.pack(pady=20, padx=20, fill='x')

file_label = ttk.Label(input_frame, text="Input File:")
file_label.pack(side='left', padx=(0, 10))

file_entry = ttk.Entry(input_frame, textvariable=file_path, width=50)
file_entry.pack(side='left', fill='x', expand=True)
file_entry.drop_target_register(DND_FILES)
file_entry.dnd_bind('<<Drop>>', drop)

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

start_button = ttk.Button(app, text="Start", command=start_processing, state='disabled')
start_button.pack(pady=(10, 20))

progress_bar = ttk.Progressbar(app, orient='horizontal', length=200, mode='determinate', maximum=100)
progress_bar.pack(pady=(0, 20))

app.mainloop()
