	
# -*- coding: utf-8 -*-
import cv2
import os
import argparse
import shutil
import numpy as np
import os
from PIL import Image
import time
import ffmpeg
import pyiqa
import torch
# print(pyiqa.list_models())
from torchvision import transforms
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import tushare as ts 
import pandas as pd 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import csv


def cal_frame(videopath):
    video_cap = cv2.VideoCapture(videopath)
    frame_count = 0
    while True:
        ret, frame = video_cap.read()
        if ret is False:
            break
        frame_count = frame_count + 1

    print(frame_count)


def cal_fps(videopath):
    video = cv2.VideoCapture(videopath)
    fps = video.get(cv2.CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    video.release()
    return fps

def get_video_info(source_video_path):
    probe = ffmpeg.probe(source_video_path)
    print('source_video_path: {}'.format(source_video_path))
    format = probe['format']
    bit_rate = int(format['bit_rate'])/1000
    duration = format['duration']
    size = int(format['size'])/1024/1024
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found!')
        return
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    num_frames = int(video_stream['nb_frames'])
    fps = int(video_stream['r_frame_rate'].split('/')[0])/int(video_stream['r_frame_rate'].split('/')[1])
    duration = float(video_stream['duration'])
    print('width: {}'.format(width))
    print('height: {}'.format(height))
    print('num_frames: {}'.format(num_frames))
    print('bit_rate: {}k'.format(bit_rate))
    print('fps: {}'.format(fps))
    print('size: {}MB'.format(size))
    print('duration: {}'.format(duration))
    return height,width,num_frames,fps

def video_to_frame(args):
    score=[]

    # create metric with default setting
    iqa_metric = pyiqa.create_metric(args.metric, device=device)
    # print(iqa_metric.lower_better)
    
    video_height,video_width,video_length,fps=get_video_info(args.video_path)
    video_capture = cv2.VideoCapture()
    video_capture.open(args.video_path)

    video_channel = 3
    transformed_image = torch.zeros([video_channel,  video_height, video_width])
    frame_idx = 1
    # video_read_index = 0
    transformations = transforms.Compose([transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    datainfo=[]
    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:

            # key frame
            # if (video_read_index < video_length):

            read_frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            read_frame = transformations(read_frame)
            transformed_image= read_frame
            transformed_image=transformed_image.unsqueeze(0)
            score_nr = iqa_metric(transformed_image)
            score_nr=score_nr.cpu()
            score_nr=score_nr.detach().numpy()
            # score_nr=score_nr[0]
            if (iqa_metric.lower_better==True)&(args.metric=='brisque'):
                score_nr=100-score_nr
                score.append(score_nr[0])
                time=frame_idx/round(fps)
                datainfo.append((frame_idx,time,score_nr[0]))
            if (iqa_metric.lower_better==True)&(args.metric=='niqe'):
                score_nr=100-score_nr
                score.append(score_nr)
                time=frame_idx/round(fps)
                datainfo.append((frame_idx,time,score_nr))
            if (args.metric=='dbcnn')|(args.metric=='cnniqa'):
                score_nr=score_nr[0]
                score.append(score_nr[0])
                time=frame_idx/round(fps)
                datainfo.append((frame_idx,time,score_nr[0]))
            # score.append(score_nr[0])
            # time=frame_idx/round(fps)
            # datainfo.append((frame_idx,time,score_nr[0]))
            frame_idx += 1





    videoname=os.path.basename(args.video_path)
    df = pd.DataFrame(datainfo,columns=['frame', 'time','score'])
    filename='frame_time_score_of_'+videoname+'.csv'
    df.to_csv(filename,index=False)
    
    ax = df.plot(x='time', y='score') 
    ax.set_xlabel('Time(s)') 
    ax.set_ylabel('Score') 
    figname='Quality_scores_of_'+videoname+'.png'
    title='Quality scores of '+videoname+' over time'
    ax.title.set_text(title)
    # setup(ax)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(round(fps)))
    # plt.ion()
    plt.savefig(figname)
    # plt.show()

    # args.threshold=int(input('Please enter a threshold: '))

    filename='Defocus_of_'+videoname+'.txt'
    Defocusindex=0
    with open(filename,'w') as f:
        for i,s in enumerate(df['score']):
            if s<args.threshold:
                print('Time of defocus: %fs, Frame of defocus: %d-th frame' % (df['time'][i], df['frame'][i]))
                f.write('Time of defocus: %fs, Frame of defocus: %d-th frame' % (df['time'][i], df['frame'][i]))
                f.write('\n')
                Defocusindex += 1
    if Defocusindex==0:
        print('The quality score of no frames is less than this threshold.')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Video_Demo")
    parser.add_argument('--video_path', type=str, default='C:\\Users\\13356\\Desktop\\AF\\VIDEO\\2.mp4')
    parser.add_argument('--metric', type=str, default='brisque',
                        help='metric: brisque|dbcnn|cnniqa|niqe')
    parser.add_argument('--threshold', type=float, default=50)
    args = parser.parse_args()
    video_to_frame(args)
#brisque,niqe(原本是lower_better，取值范围是0-100，使用100减去原始分数，变成high_better)