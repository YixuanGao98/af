import os
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
import numpy as np
import random
import cv2

from torchvision import transforms
from models.maniqa import MANIQA
from torch.utils.data import DataLoader
from config import Config
# from utils.inference_process import ToTensor, Normalize
# from tqdm import tqdm

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def video_processing(frame, num_crops, transform=None):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.array(frame).astype('float32') / 255
    h, w, c = frame.shape

    if transform:
        frame = transform(frame)
    new_h = 224
    new_w = 224

    img_patches = torch.zeros([num_crops, 3, new_h, new_w])
    for i in range(num_crops):
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        patch = frame[:, top: top + new_h, left: left + new_w]
        img_patches[i] = patch
    return img_patches

if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(20)

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--videopath', type=str,
                        default='/home/cyq/Work/Project/Huawei/Light-VQA/data/my_dataset/31_MBLLEN.mp4')
    args = parser.parse_args()

    # config file
    config = Config({
        # image path
        "video_path": args.videopath,
        #00021.mp4 score: 0.59081452836593
        #00021_AGCCPF.mp4 score: 0.5601685643196106
        #00021_BPHEME.mp4 score: 0.5579883915682634
        #00021_GHE.mp4 score: 0.4677622988820076

        # valid times
        "frame_interval": 10,
        "num_crops": 20,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,

        # checkpoint path
        "ckpt_path": "./mode.pth",
    })


    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
                 patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
                 depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)
    net.load_state_dict(torch.load(config.ckpt_path,map_location=device))
    # net = torch.load('./ckpt_valid')
    net = net.to(device)

    avg_score = 0
    cap = cv2.VideoCapture(config.video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length_read = int(video_length / config.frame_interval)
    transform_frame = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    video_read_index = 0
    with torch.no_grad():
        net.eval()
        for i in range(video_length):
            has_frames, frame = cap.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length_read) and (i % config.frame_interval == 0):
                    patchs = video_processing(frame, config.num_crops, transform_frame)
                    patchs = patchs.to(device)
                    score = net(patchs)
                    avg_score += torch.mean(score).to('cpu').numpy()
                    # print(torch.mean(score).to('cpu').numpy())
                    video_read_index += 1
            else:
                break


    final_score = avg_score / video_read_index
    # final_score = avg_score.to('cpu').numpy()
    print("{} : {}".format(config.video_path, final_score))

