import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import sys


def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def process_images(model, images_tensor):
  model.feed_data(data={'lq': images_tensor})
  model.test()
  visuals = model.get_current_visuals()
  out_imgs = []
  for out_tensor in visuals['result']:
      out_img = tensor2img(out_tensor)
      final_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
      out_imgs.append(final_img)
  return out_imgs

def process_video(model, video_path, output_folder, batch_size, progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    base_filename = os.path.basename(video_path)
    output_filename = os.path.splitext(base_filename)[0] + "_deblured.mp4"
    output_path = os.path.join(output_folder, output_filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    batch_frames = []
    processed_frames = 0  # Track the number of processed frames
    start_time = time.time()  # Start time for ETA and FPS calculations
    total_proc_time = 0  # Accumulator for total processing time

    print("Started processing video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor = torch.unsqueeze(img2tensor(frame), 0)
        batch_frames.append(frame_tensor)

        if len(batch_frames) == batch_size:
            frame_proc_start = time.time()  # Time when batch processing starts
            batch_tensor = torch.cat(batch_frames, dim=0)
            processed_images = process_images(model, batch_tensor)
            batch_proc_time = time.time() - frame_proc_start
            total_proc_time += batch_proc_time  # Add current batch processing time to total

            del batch_tensor  # Free memory immediately after usage
            for img in processed_images:
                out.write(img)
            processed_frames += len(batch_frames)
            if progress_callback:
                progress_callback(processed_frames, total_frames)
            
            # Calculate and display average FPS and update the progress and ETA
            average_fps = processed_frames / total_proc_time if total_proc_time > 0 else float('inf')
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / processed_frames * total_frames
            eta = estimated_total_time - elapsed_time
            progress = processed_frames / total_frames * 100
            sys.stdout.write(f"\rProgress: {progress:.2f}% | ETA: {eta:.2f}s | Average FPS: {average_fps:.2f}")
            sys.stdout.flush()
            batch_frames = []

    if batch_frames:
        frame_proc_start = time.time()
        batch_tensor = torch.cat(batch_frames, dim=0)
        processed_images = process_images(model, batch_tensor)
        batch_proc_time = time.time() - frame_proc_start
        total_proc_time += batch_proc_time  # Include remaining frames in total time

        for img in processed_images:
            out.write(img)
        processed_frames += len(batch_frames)
        if progress_callback:
            progress_callback(processed_frames, total_frames)

    average_fps = processed_frames / total_proc_time if total_proc_time > 0 else float('inf')

    cap.release()
    out.release()
    print(f"\nVideo processing complete. Final Average Processing FPS: {average_fps:.2f}")