""" ************************************************
* fileName: bsd.py
* desc: Beam Splitter Deblurring dataset for real-world video deblurring.
* author: mingdeng_cao
* date: 2021/12/04 16:46
* last revised: None
************************************************ """


import os
import sys
import platform

import torch
import numpy as np
import cv2
from .augment import augment

from .build import DATASET_REGISTRY

import logging


@DATASET_REGISTRY.register()
class BSD(torch.utils.data.Dataset):
    """
    Args:
        cfg(Easydict): The config file for dataset. 
            root_gt(str): the root path of gt videos
            root_input(str): the root path of the input videos

    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.video_list = os.listdir(self.cfg.root_gt)
        self.video_list.sort()

        self.frames = []
        self.video_frame_dict = {}
        self.video_length_dict = {}

        self.total_num_frames = 0
        for video_name in self.video_list:
            # Warning! change the video path in different format of video deblurring dataset
            video_path = os.path.join(self.cfg.root_gt, video_name, "Sharp", "RGB")
            frames_in_video = os.listdir(video_path)
            frames_in_video.sort()
            self.total_num_frames += len(frames_in_video)
            frames_in_video = [os.path.join(
                video_name, frame) for frame in frames_in_video]

            # sample length with inerval
            sampled_frames_length = (cfg.num_frames - 1) * cfg.interval + 1

            if cfg.sampling == "n_n" or cfg.sampling == "n_l":
                # non-overlapping sampling
                if cfg.overlapping:
                    # avoid  1 - sampled_frames_length = 0, transfer it to positive index
                    self.frames += frames_in_video[:len(
                        frames_in_video) - sampled_frames_length + 1]
                else:
                    # ensure the sampling frame can be sampled!
                    self.frames += frames_in_video[:len(
                        frames_in_video) - sampled_frames_length + 1:sampled_frames_length]

            elif cfg.sampling == "n_c":
                if cfg.overlapping:
                    self.frames += frames_in_video[sampled_frames_length // 2: len(
                        frames_in_video) - (sampled_frames_length // 2)]
                else:
                    self.frames += frames_in_video[sampled_frames_length // 2: len(
                        frames_in_video) - (sampled_frames_length // 2): sampled_frames_length]

            elif cfg.sampling == "n_r":
                if cfg.overlapping:
                    self.frames += frames_in_video[sampled_frames_length-1:]
                else:
                    self.frames += frames_in_video[sampled_frames_length -
                                                   1::sampled_frames_length]

            # strcnn style sampling
            elif cfg.sampling == "strcnn":
                assert len(
                    frames_in_video) > 4, "The num of frames in the video "
                if cfg.overlapping:
                    self.frames += frames_in_video[2: len(
                        frames_in_video) - (cfg.num_frames - 2) + 1]
                else:
                    self.frames += frames_in_video[2: len(frames_in_video) - (
                        cfg.num_frames - 2) + 1: cfg.num_frames]

            # you can add some other sampling mode here.
            else:
                print("none sampling mode '{}' ".format(cfg.sampling))
                raise NotImplementedError

            self.video_frame_dict[video_name] = frames_in_video
            self.video_length_dict[video_name] = len(frames_in_video)

        assert self.frames, "Their is no frames in '{}'. ".format(
            self.cfg.root_gt)

        # print(self.frames)
        logger = logging.getLogger("simdeblur")
        logger.info(f"Dataset Name: {cfg.name} for {cfg.mode}")
        logger.info(f"GT Root: {cfg.root_gt}")
        logger.info(f"Total Frames: {self.total_num_frames}")
        logger.info(f"Sampled Mode: {cfg.sampling}")
        logger.info(f"Valid Samples: {len(self.frames)}")

    def __getitem__(self, idx):
        if platform.system() == "Windows":
            video_name, frame_name = self.frames[idx].split("\\")
        else:
            video_name, frame_name = self.frames[idx].split("/")
        frame_idx, suffix = frame_name.split(".")
        frame_idx_length = len(frame_idx)
        frame_idx = int(frame_idx)
        video_length = self.video_length_dict[video_name]
        # print("video: {} frame: {}".format(video_name, frame_idx))

        gt_frames_name = [frame_name]
        input_frames_name = []

        # when to read the frames, should pay attention to the name of frames
        frame_name_format = "{:0" + str(frame_idx_length) + "d}.{}"
        if self.cfg.sampling == "n_c":
            input_frames_name = [frame_name_format.format(i, suffix) for i in range(
                frame_idx - (self.cfg.num_frames // 2) * self.cfg.interval, frame_idx + (self.cfg.num_frames // 2) * self.cfg.interval + 1, self.cfg.interval)]

        elif self.cfg.sampling == "n_n" or self.cfg.sampling == "n_l":
            input_frames_name = [frame_name_format.format(i, suffix) for i in range(
                frame_idx, frame_idx + self.cfg.interval * self.cfg.num_frames, self.cfg.interval)]
            if self.cfg.sampling == "n_n":
                gt_frames_name = [frame_name_format.format(i, suffix) for i in range(
                    frame_idx, frame_idx + self.cfg.interval * self.cfg.num_frames, self.cfg.interval)]

        elif self.cfg.sampling == "n_r":
            input_frames_name = [frame_name_format.format(i, suffix) for i in range(
                frame_idx - self.cfg.num_frames * self.cfg.interval + 1, frame_idx + 1, self.cfg.interval)]

        elif self.cfg.sampling == "strcnn":
            gt_frames_name = [frame_name_format.format(i, suffix) for i in range(
                frame_idx, frame_idx + self.cfg.num_frames - 4)]
            input_frames_name = [frame_name_format.format(i, suffix) for i in range(
                frame_idx - 2, frame_idx + self.cfg.num_frames - 2)]

        else:
            raise NotImplementedError

        assert len(input_frames_name) == self.cfg.num_frames, "Wrong frames length not equal the sampling frames {}".format(
            self.cfg.num_frames)

        # Warning! Chaning the path of different deblurring datasets.
        gt_frames_path = os.path.join(self.cfg.root_gt, video_name, "Sharp", "RGB", "{}")
        input_frames_path = os.path.join(self.cfg.root_gt, video_name, "Blur", "RGB", "{}")

        # Read images by opencv with format HWC, BGR, [0,1], TODO add other loading methods.
        gt_frames = [read_img_opencv(gt_frames_path.format(
            frame_name)) for frame_name in gt_frames_name]
        input_frames = [read_img_opencv(input_frames_path.format(
            frame_name)) for frame_name in input_frames_name]

        # stack and transpose with RGB style (n, c, h, w)
        gt_frames = np.stack(
            gt_frames, axis=0)[..., ::-1].transpose([0, 3, 1, 2])
        input_frames = np.stack(
            input_frames, axis=0)[..., ::-1].transpose([0, 3, 1, 2])

        # augmentaion while training...
        if self.cfg.mode == "train" and hasattr(self.cfg, "augmentation"):
            input_frames, gt_frames = augment(
                input_frames, gt_frames, self.cfg.augmentation)

        # To tensor with contingious array.
        gt_frames = torch.tensor(np.ascontiguousarray(gt_frames)).float()
        input_frames = torch.tensor(np.ascontiguousarray(input_frames)).float()

        return {
            "input_frames": input_frames,
            "gt_frames": gt_frames,
            "video_name": video_name,
            "video_length": video_length,
            "gt_names": gt_frames_name,
        }

    def __len__(self):
        return len(self.frames)


def read_img_opencv(path, size=None):
    """
    read image by opencv
    return: Numpy float32, HWC, BGR, [0,1]
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("the path is None! {} !".format(path))
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img
