import cv2
import os
import argparse
import shutil
import numpy as np
# References: https://blog.csdn.net/bryant_meng/article/details/110079285
# Check test_video.sh in the main directory
# Step1: Run video_to_frame to convert low light video to images
# Step2: Run test.py to convert low light images to enhanced images
# Step3: Run image_to_video to convert enhanced images to enhanced videos
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Video_Demo")
    parser.add_argument('--video_path', type=str, default='/Users/apple/Desktop/image_enhancement/database/video4/')
    # parser.add_argument("--image_lowlight_folder", type=str, default='/home/gyx/image_enhancement/database/frame/00039/%d.jpg')
    parser.add_argument("--image_lowlight_folder", type=str, default='/Users/apple/Desktop/image_enhancement/database/resized_frame4/')
    parser.add_argument('--image_Enhancefolder', type=str, default='/Users/apple/Desktop/image_enhancement/database/resized_frame3/')
    # parser.add_argument('--save_path', type=str, default='demo/Movie/Res.mp4')
    parser.add_argument('--save_path', type=str, default='/Users/apple/Desktop/image_enhancement/database/video3/')

    parser.add_argument('--choice', type=str, choices = ['V2I', 'I2V'], default='V2I')

    args = parser.parse_args()
    return args


def cal_frame(args):
    video_cap = cv2.VideoCapture(args.video_path)
    frame_count = 0
    while True:
        ret, frame = video_cap.read()
        if ret is False:
            break
        frame_count = frame_count + 1

    print(frame_count)


def cal_fps(args):
    video = cv2.VideoCapture(videopath)
    fps = video.get(cv2.CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    video.release()
    return fps


def video_to_frame(videopath,outpath):
    vidcap = cv2.VideoCapture(videopath) # TODO: replace with url like youtube link
    success, image = vidcap.read()
    outpath=outpath+'%d.jpg'
    # image=np.resize(image,(720,1280))
    

    count = 0
    while success:
        cv2.imwrite(outpath % count, image)
        img = Image.open(outpath % count)
        out = img.resize((1280, 720),Image.ANTIALIAS)
        out.save(outpath % count)

        success, image = vidcap.read()
        # image=np.resize(image,(720,1280))
        
        
        print('Read a new frame: ', success)
        count += 1
        print("We have %2d images" % count)


def image_to_video(videopath,outpath,save_Enhancedpath):
    img = cv2.imread(outpath + '0.jpg')
    fps = cal_fps(videopath)
    size = (img.shape[1], img.shape[0])
    print(size)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # For mp4 only
    
    videoWrite = cv2.VideoWriter(save_Enhancedpath, fourcc, fps, size)

    files = os.listdir(outpath)
    out_num = len(files)

    for i in range(0, out_num):
        fileName = outpath+ str(i) + '.jpg'
        img = cv2.imread(fileName)
        videoWrite.write(img)

def main(videopath,outpath,save_Enhancedpath):
    if args.choice == 'V2I':
        video_to_frame(videopath,outpath)
    elif args.choice == 'I2V':
        image_to_video(videopath,outpath,save_Enhancedpath)
    else:
        raise TypeError

if __name__ == "__main__":
    # method=['DCC-Net','ACE','HDR','AGCCPF','MBLLEN']
    method=['HDR']
    args = parse_args()
    if args.choice == 'V2I':
        allvideo_names = os.listdir(args.video_path)
        for video_name in allvideo_names:
            videopath =args.video_path+video_name

            name=video_name.split('.')
            name=name[0]

            outpath=args.image_lowlight_folder+name+'/'

            if os.path.exists(outpath) and os.path.isdir(outpath):
                shutil.rmtree(outpath)
            os.makedirs(outpath)
            main(videopath,outpath,args.save_path)
    elif args.choice == 'I2V':
        allvideo_names = os.listdir(args.video_path)
        for video_name in allvideo_names:
            if video_name[0]!='.':
                videopath =args.video_path+video_name
                name=video_name.split('.')
                name=name[0]
                for m in method:
                    outpath_enhanced=args.image_Enhancefolder+m+'/'+name+'/'

                    save_path=args.save_path+name+'_'+m+'.mp4'##resize
                    # if name=='00042':
                    main(videopath,outpath_enhanced,save_path)


