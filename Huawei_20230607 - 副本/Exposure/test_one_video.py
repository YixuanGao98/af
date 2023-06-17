# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
# import torch.optim as optim
# from utils import performance_fit
# from utils import L1RankLoss
# import torch.nn as nn

# from my_dataloader import VideoDataset_spatio_temporal_brightness
from final_fusion_model import swin_small_patch4_window7_224 as create_model
from extract_temporal_features import temporal_feature

from torchvision import transforms
# import time
import cv2
from PIL import Image
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def video_processing(video_path, device):
    cap = cv2.VideoCapture(video_path)
    video_channel = 3
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_length_read = int(video_length / video_frame_rate)

    ## spatial
    transformed_video = torch.zeros([video_length_read, video_channel, 720, 1280])
    transformations_test = transforms.Compose(
        [transforms.Resize(720), transforms.CenterCrop([720, 1280]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## brightness
    brightness = torch.zeros([video_frame_rate * video_length_read, 144])
    transform_brightness = transforms.Compose(
        [transforms.Resize([720, 1280]), transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])

    ## temporal
    resize = 224
    transform_temporal = transforms.Compose([transforms.Resize([resize, resize]), transforms.ToTensor(), \
                                          transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                               std=[0.225, 0.225, 0.225])])
    temporal_video = torch.zeros([video_length, video_channel, resize, resize])

    video_read_index = 0
    frame_idx = 0
    for i in range(video_length):
        has_frames, frame = cap.read()
        if has_frames:
            # key frame
            if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == 0):
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                ## brightness
                if i < video_frame_rate * video_length_read:
                    frame_brightness = frame.convert('L')  # 转gray
                    frame_brightness = transform_brightness(frame_brightness)
                    frame_brightness = torch.reshape(frame_brightness, [144, 80, 80])
                    frame_brightness = torch.flatten(frame_brightness, start_dim=1)
                    brightness_mean = torch.mean(frame_brightness, dim=1)
                    brightness[i] = brightness_mean

                ## temporal
                frame_temporal = transform_temporal(frame)
                temporal_video[video_read_index] = frame_temporal

                ## spatial data
                frame_spatial = transformations_test(frame)
                transformed_video[video_read_index] = frame_spatial
                video_read_index += 1
            frame_idx += 1
        else:
            break

    if video_read_index < video_frame_rate * video_length_read:
        for i in range(video_read_index, video_frame_rate * video_length_read):
            brightness[i] = brightness[video_read_index - 1]
    if video_read_index < video_length:
        for i in range(video_read_index, video_length):
            temporal_video[i] = temporal_video[video_read_index - 1]
    if video_read_index < video_length_read:
        for i in range(video_read_index, video_length_read):
            transformed_video[i] = transformed_video[video_read_index - 1]

    brightness = torch.reshape(brightness, [video_length_read, video_frame_rate, 144])
    brightness_consistency = torch.var(brightness, dim=1)
    brightness_consistency *= 10
    transformed_feature = temporal_feature(temporal_video, video_length, video_frame_rate, device)
    transformed_feature = torch.cat([transformed_feature,brightness_consistency], dim=1)
    # transformed_video = transformed_video.to(device)

    return transformed_video, transformed_feature



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)

    weights = 'ckpts/last2_SI+TI_epoch_16_SRCC_0.937369.pth'
    weights_dict = torch.load(weights, map_location=device)
    print(model.load_state_dict(weights_dict))
    with torch.no_grad():
        model.eval()
        video, tem_f = video_processing(args.videopath, device)
        video = video.to(device)
        tem_f = tem_f.to(device)
        outputs = model(video, tem_f)
        y_output = torch.mean(outputs).item()
        print("{} : {}" .format(args.videopath, y_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--videopath', type=str, default='/home/cyq/Work/Project/Huawei/Light-VQA/data/my_dataset/31_MBLLEN.mp4')
    # 00021.mp4 31.668933868408203
    # 00021_BPHEME.mp4 30.637622833251953
    # 00021_DCC-Net.mp4 35.27886199951172
    # 00021_AGCCPF.mp4 36.236812591552734
    # 00021_JY.mp4 39.06159591674805
    # 00021_SGZSL.mp4 45.23496627807617
    # 00021_MBLLEN.mp4 33.10592269897461
    # 00021_StableLLVE.mp4 47.81495666503906
    # 00021_GHE.mp4 44.32536315917969
    args = parser.parse_args()
    main(args)