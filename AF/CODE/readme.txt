
运行：sh run.sh
参数：
video_path：视频路径
metric：使用的IQA方法，包括brisque|dbcnn|cnniqa|niqe
threshold：阈值，低于该分数的视频的帧数会被记录下来

保存文件：
frame_time_score_of_xx.csv:视频每一帧的质量分数
Quality_scores_of_xx.png:视频帧随时间变化的质量分数
Defocus_of_xx.txt：低于阈值的视频的帧数会被记录下来
