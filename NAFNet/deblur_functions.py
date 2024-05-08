import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_tensor = torch.unsqueeze(img2tensor(frame), 0)
        batch_frames.append(frame_tensor)

        if len(batch_frames) == batch_size:
            batch_tensor = torch.cat(batch_frames, dim=0)
            processed_images = process_images(model, batch_tensor)
            for img in processed_images:
                out.write(img)
            processed_frames += len(batch_frames)
            if progress_callback:
                progress_callback(processed_frames, total_frames)
            batch_frames = []

    if batch_frames:
        batch_tensor = torch.cat(batch_frames, dim=0)
        processed_images = process_images(model, batch_tensor)
        for img in processed_images:
            out.write(img)
        processed_frames += len(batch_frames)
        if progress_callback:
            progress_callback(processed_frames, total_frames)

    cap.release()
    out.release()
