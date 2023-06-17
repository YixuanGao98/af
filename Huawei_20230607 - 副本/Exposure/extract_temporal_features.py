# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch

from my_dataloader import VideoDataset_extract_temporal_feature

from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn
import cv2
from PIL import Image

def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list


class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0, 5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)

            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)

        return slow_feature, fast_feature

def temporal_feature(transformed_frame_all, video_length, video_frame_rate, device):
    model = slowfast()

    model = model.to(device)

    ## load video clip
    resize = 224
    # transformations = transforms.Compose([transforms.Resize([resize, resize]), transforms.ToTensor(), \
    #                                       transforms.Normalize(mean=[0.45, 0.45, 0.45],
    #                                                            std=[0.225, 0.225, 0.225])])
    # cap = cv2.VideoCapture(video_path)
    video_channel = 3
    #
    # video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    video_clip = int(video_length / video_frame_rate)
    video_clip_min = 8
    video_length_clip = 32

    # transformed_frame_all = torch.zeros([video_length, video_channel, resize, resize])
    #
    #
    #
    # video_read_index = 0
    # for i in range(video_length):
    #     has_frames, frame = cap.read()
    #     if has_frames:
    #         read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #         read_frame = transformations(read_frame)
    #         transformed_frame_all[video_read_index] = read_frame
    #         video_read_index += 1
    #
    # if video_read_index < video_length:
    #     for i in range(video_read_index, video_length):
    #         transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
    #
    # cap.release()

    transformed_video_all = []
    for i in range(video_clip):
        transformed_video = torch.zeros([video_length_clip, video_channel, resize, resize])
        if (i * video_frame_rate + video_length_clip) <= video_length:
            transformed_video = transformed_frame_all[
                                i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
        else:
            transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
            for j in range((video_length - i * video_frame_rate), video_length_clip):
                transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
        transformed_video_all.append(transformed_video)

    if video_clip < video_clip_min:
        for i in range(video_clip, video_clip_min):
            transformed_video_all.append(transformed_video_all[video_clip - 1])

    feature_motion = torch.zeros([len(transformed_video_all), 2048 + 256])
    with torch.no_grad():
        model.eval()
        for idx, ele in enumerate(transformed_video_all):
            ele = ele.unsqueeze(dim=0)
            ele = ele.permute(0, 2, 1, 3, 4)
            ele = pack_pathway_output(ele, device)
            ele_slow_feature, ele_fast_feature = model(ele)

            ele_slow_feature = ele_slow_feature.squeeze()
            ele_fast_feature = ele_fast_feature.squeeze()

            ele_feature_motion = torch.cat([ele_slow_feature, ele_fast_feature])
            ele_feature_motion = ele_feature_motion.unsqueeze(dim=0)

            feature_motion[idx] = ele_feature_motion

    return feature_motion


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = slowfast()

    model = model.to(device)

    resize = args.resize

    ## training data
    # videos_dir = 'D:/second_semester/LIVE-VQC/my_dataset'
    # datainfo_train = 'data/my_train.csv'
    # datainfo_test = 'data/my_test.csv'
    llv_dir = 'D:/second_semester/cross_dataset/llv_720'
    datainfo_llv = 'data/llv_konvid.csv'
    transformations = transforms.Compose([transforms.Resize([resize, resize]), transforms.ToTensor(), \
                                               transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                                    std=[0.225, 0.225, 0.225])])
    # trainset = VideoDataset_extract_temporal_feature(videos_dir, datainfo_train, transformations, resize)
    # testset = VideoDataset_extract_temporal_feature(videos_dir, datainfo_test, transformations, resize)
    llv_set = VideoDataset_extract_temporal_feature(llv_dir, datainfo_llv, transformations, resize)

    ## dataloader
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
    #                                            shuffle=False, num_workers=args.num_workers)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
    #                                           shuffle=False, num_workers=args.num_workers)
    llv_loader = torch.utils.data.DataLoader(llv_set, batch_size=1,
                                               shuffle=False, num_workers=args.num_workers)
    # do validation after each epoch
    with torch.no_grad():
        model.eval()
        # for i, (video, mos, video_name) in enumerate(train_loader):
        #     video_name = video_name[0]
        #     print(video_name)
        #     if not os.path.exists(args.feature_save_folder + '/' + video_name.split('.')[0]):
        #         os.makedirs(args.feature_save_folder + video_name.split('.')[0])
        #
        #     for idx, ele in enumerate(video):
        #         # ele = ele.to(device)
        #         ele = ele.permute(0, 2, 1, 3, 4)
        #         inputs = pack_pathway_output(ele, device)
        #         slow_feature, fast_feature = model(inputs)
        #         np.save(args.feature_save_folder + video_name.split('.')[0] + '/' + 'feature_' + str(idx) + '_slow_feature',
        #                 slow_feature.to('cpu').numpy())
        #         np.save(args.feature_save_folder + video_name.split('.')[0] + '/' + 'feature_' + str(idx) + '_fast_feature',
        #                 fast_feature.to('cpu').numpy())

        # for i, (video, mos, video_name) in enumerate(test_loader):
        #     video_name = video_name[0]
        #     print(video_name)
        #     if not os.path.exists(args.feature_save_folder + video_name.split('.')[0]):
        #         os.makedirs(args.feature_save_folder + video_name.split('.')[0])
        #
        #     for idx, ele in enumerate(video):
        #         # ele = ele.to(device)
        #         ele = ele.permute(0, 2, 1, 3, 4)
        #         inputs = pack_pathway_output(ele, device)
        #         slow_feature, fast_feature = model(inputs)
        #         np.save(args.feature_save_folder + video_name.split('.')[0] + '/' + 'feature_' + str(idx) + '_slow_feature',
        #                 slow_feature.to('cpu').numpy())
        #         np.save(args.feature_save_folder + video_name.split('.')[0] + '/' + 'feature_' + str(idx) + '_fast_feature',
        #                 fast_feature.to('cpu').numpy())

        for i, (video, mos, video_name) in enumerate(llv_loader):
            video_name = video_name[0]
            print(video_name)
            if not os.path.exists(args.feature_save_folder + video_name.split('.')[0]):
                os.makedirs(args.feature_save_folder + video_name.split('.')[0])

            for idx, ele in enumerate(video):
                # ele = ele.to(device)
                ele = ele.permute(0, 2, 1, 3, 4)
                inputs = pack_pathway_output(ele, device)
                slow_feature, fast_feature = model(inputs)
                np.save(args.feature_save_folder + video_name.split('.')[0] + '/' + 'feature_' + str(idx) + '_slow_feature',
                        slow_feature.to('cpu').numpy())
                np.save(args.feature_save_folder + video_name.split('.')[0] + '/' + 'feature_' + str(idx) + '_fast_feature',
                        fast_feature.to('cpu').numpy())
                if idx == 7:
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str, default='SlowFast')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--feature_save_folder', type=str, default='llv_konvid_temporal_feature/')

    args = parser.parse_args()

    main(args)