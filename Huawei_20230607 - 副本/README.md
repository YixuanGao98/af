# Video Exposure
输入一个视频文件路径得出曝光度分数
```
cd Exposure
python test_one_video.py -v $YOUR_SPECIFIED_VIDEO_PATH$
```

# Video Color
输入一个视频文件路径得出色彩分数
```
cd Color
python test_one_video.py -v $YOUR_SPECIFIED_VIDEO_PATH$
```
# Video AF
```
cd AF/CODE
运行：sh run.sh
参数：
video_path：视频路径
metric：使用的IQA方法，包括brisque|dbcnn|cnniqa|niqe
threshold：阈值，低于该分数的视频的帧数会被记录下来

保存文件：
frame_time_score_of_xx.csv:视频每一帧的质量分数
Quality_scores_of_xx.png:视频帧随时间变化的质量分数
Defocus_of_xx.txt：低于阈值的视频的帧数会被记录下来
```

# Video Noise
输入一个视频文件路径得出噪声分数
```
cd Noise
python test_one_video.py -v $YOUR_SPECIFIED_VIDEO_PATH$
```