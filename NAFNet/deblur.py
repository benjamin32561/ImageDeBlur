import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import imageio
from PIL import Image

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def display(img1, img2):
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title('Input image', fontsize=16)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.title('NAFNet output', fontsize=16)
    ax2.axis('off')
    ax1.imshow(img1)
    ax2.imshow(img2)

def create_difference_gif(image1_path, image2_path, gif_path, num_transitions=5, line_width=5):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    
    frames = []
    width = img1_array.shape[1]
    step_size = width // num_transitions
    
    for i in range(num_transitions + 1):
        current_width = i * step_size
        if current_width > width:
            current_width = width
        
        new_image = np.copy(img1_array)
        
        if current_width + line_width < width:
            new_image[:, current_width:current_width + line_width] = [255, 255, 255]
        
        if current_width + line_width < width:
            new_image[:, current_width + line_width:] = img2_array[:, current_width + line_width:]
        
        new_image_pil = Image.fromarray(new_image)
        frames.append(new_image_pil)
    
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=1000, loop=0)


def single_image_inference(model, img, save_path):
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})
    model.test()
    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, save_path)

def process_images(input_path, output_folder, NAFNet, create_gif = True):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    else:
        files = [input_path]
    for file in files:
        img_input = imread(file)
        inp = img2tensor(img_input)
        output_path = os.path.join(output_folder, os.path.basename(file))
        single_image_inference(NAFNet, inp, output_path)
        img_output = imread(output_path)
        if create_gif:
            gif_path = os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0] + '_diff.gif')
            create_difference_gif(output_path, file, gif_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_path> <output_folder> <opt_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_folder = sys.argv[2]
    opt_path = sys.argv[3]

    # Load model configuration and model
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    NAFNet = create_model(opt)

    process_images(input_path, output_folder, NAFNet)