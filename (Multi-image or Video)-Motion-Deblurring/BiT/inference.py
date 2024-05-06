import yaml
import os
import torch
import time
import importlib
import cv2
import os.path as osp
import torch.distributed as dist
import numpy as np
import imageio as iio
from tqdm import tqdm
from argparse import ArgumentParser
from einops import rearrange
from model.utils import init_seeds

if __name__ == '__main__':
    """
    python inference.py --config ./checkpoints/bit_adobe240/cfg.yaml --checkpoint ./checkpoints/bit_adobe240/latest.ckpt --img_pre ./demo/00777.png --img_cur ./demo/00785.png --img_nxt ./demo/00793.png --save_dir ./demo/bit_results/ 
    """
    parser = ArgumentParser(description='blur interpolation transformer')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--config', default='./configs/cfg.yaml', help='path of config')
    parser.add_argument('--checkpoint', type=str, default=None, help='path of checkpoint')
    parser.add_argument('--img_pre', type=str, required=True, help='path of previous blurry image')
    parser.add_argument('--img_cur', type=str, required=True, help='path of current blurry image')
    parser.add_argument('--img_nxt', type=str, default=None, help='path of next blurry image')
    parser.add_argument('--save_dir', type=str, required=True, help='where to save image results')
    parser.add_argument('--num', type=int, default=1, help='number of extracted images')
    parser.add_argument('--gif', type=bool, default=False, help='whether to generate the corresponding gif')
    parser.add_argument('--multi_infer', type=bool, default=True, help='multiple inferences with shared features')
    args = parser.parse_args()

    # parse cfgs
    with open(args.config) as f:
        cfgs = yaml.full_load(f)

    # Device initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # CUDA specific initializations if CUDA is available
    if device.type == 'cuda':
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True

    init_seeds(seed=args.local_rank)

    # Distributed Data Parallel (DDP) Initialization
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")

    # create model
    model_cls = getattr(importlib.import_module('model'), cfgs['model_args']['name'])

    model = model_cls(**cfgs['model_args']['args'],
                    optimizer_args=cfgs['optimizer_args'],
                    scheduler_args=cfgs['scheduler_args'],
                    loss_args=cfgs['loss_args'],
                    local_rank=args.local_rank,
                    load_from=args.checkpoint)


    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Inference and saving results
    eval_time_interval = 0
    ts = torch.linspace(start=0, end=1, steps=args.num, device=device)
    img_pre = cv2.imread(args.img_pre)
    img_cur = cv2.imread(args.img_cur)
    lq_imgs = [img_pre, img_cur]
    if args.img_nxt is not None:
        img_nxt = cv2.imread(args.img_nxt)
        lq_imgs.append(img_nxt)

    save_name = osp.join(save_dir, 'blur.png')
    cv2.imwrite(save_name, img_cur)
    gif_imgs = 15 * [img_cur, ]
    lq_imgs = np.stack(lq_imgs, axis=0)[np.newaxis]  # 1, 2, H, W, C
    lq_imgs = torch.from_numpy(lq_imgs).to(device, non_blocking=True).float() / 255.
    lq_imgs = rearrange(lq_imgs, 'b n h w c -> b n c h w')

    if not args.multi_infer:
        for i, t in tqdm(enumerate(ts), total=args.num):
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            time_stamp = time.time()
            pred_img = model.inference(lq_imgs, [t])
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            eval_time_interval += time.time() - time_stamp
            pred_img = pred_img.detach().clamp(0, 1)[0][0]
            pred_img = rearrange(pred_img * 255., 'c h w -> h w c').cpu().detach().numpy().astype(np.uint8)
            save_name = osp.join(save_dir, '{:08d}.png'.format(i))
            cv2.imwrite(save_name, pred_img)
            gif_imgs.append(pred_img)
    else:
        if model.device.type == 'cuda':
            torch.cuda.synchronize()
        time_stamp = time.time()
        pred_imgs = model.inference(lq_imgs, ts)
        if model.device.type == 'cuda':
            torch.cuda.synchronize()
        eval_time_interval += time.time() - time_stamp
        pred_imgs = pred_imgs.detach().clamp(0, 1)[0]
        for i in range(args.num):
            pred_img = pred_imgs[i]
            pred_img = rearrange(pred_img * 255., 'c h w -> h w c').cpu().detach().numpy().astype(np.uint8)
            save_name = osp.join(save_dir, '{:08d}.png'.format(i))
            cv2.imwrite(save_name, pred_img)
            gif_imgs.append(pred_img)
    if args.gif:
        gif_path = osp.join(args.save_dir, 'demo.gif')
        with iio.get_writer(gif_path, mode='I') as writer:
            for img in gif_imgs:
                writer.append_data(img[:, :, ::-1])

    print('total runtime: {:.3f}'.format(eval_time_interval))
    print('average runtime: {:.3f}'.format(eval_time_interval / args.num))
    
    if torch.cuda.is_available():
        dist.destroy_process_group()