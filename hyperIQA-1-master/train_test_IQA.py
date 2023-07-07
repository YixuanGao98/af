import os
import argparse
import random
import numpy as np
from HyerIQASolver import HyperIQASolver


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main(config):

    folder_path = {
        'live': '/DATA/gaoyixuan_data/imagehist/LIVE/',
        'csiq': '/home/gyx/DATA/imagehist/CSIQ/',
        'TID': '/DATA/gaoyixuan_data/tid2013/',
        'livec': '/DATA/gaoyixuan_data/imagehist/CLIVE/',
        'koniq-10k': '/home/gyx/DATA/imagehist/KON',
        'LIVEMD': '/DATA/gaoyixuan_data/imagehist/LIVEMD/',
        'cid': '/DATA/gaoyixuan_data/imagehist/CID2013'
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'TID': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'cid': list(range(0, 474)),
        'LIVEMD': list(range(0, 15)),
    }
    sel_num = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)
    rmse_all = np.zeros(config.train_test_num, dtype=np.float)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        # Randomly select 80% images for training and the rest for testing
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index)
        srcc_all[i], plcc_all[i], rmse_all[i] = solver.train()
        print('Testing  SRCC %4.4f,\t PLCC %4.4f,\t rmse %4.4f' % (srcc_all[i], plcc_all[i],rmse_all[i]))

    # print(srcc_all)
    # print(plcc_all)
    srcc_med = np.mean(srcc_all)
    plcc_med = np.mean(plcc_all)
    rmse_med = np.mean(rmse_all)

    print('Testing  SRCC %4.4f,\t PLCC %4.4f,\t rmse %4.4f' % (srcc_med, plcc_med,rmse_med))

    return srcc_med, plcc_med,rmse_med


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='cid', help='Support datasets: livec|koniq-10k|LIVEMD|live|csiq|tid2013|cid')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=18, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')

    config = parser.parse_args()

    srcc_all = np.zeros((1,10),dtype=np.float)
    plcc_all = np.zeros((1,10),dtype=np.float)
    rmse_all = np.zeros((1,10),dtype=np.float)

    for i in range(0,10):
        srcc, plcc,rmse=main(config)

        srcc_all[0][i] = srcc
        plcc_all[0][i] = plcc
        rmse_all[0][i] = rmse

    srcc_mean = np.mean(srcc_all)
    plcc_mean = np.mean(plcc_all)
    rmse_mean = np.mean(rmse_all)

    print(srcc_all)
    print('average MOSsrcc:%4.4f' % (srcc_mean))  
    print(plcc_all)
    print('average MOSplcc:%4.4f' % (plcc_mean))  
    print(rmse_all)
    print('average MOSrmse:%4.4f' % (rmse_mean))  
    print('hyperiqa-CID')
######1遍：：
# KON:4       0.086           0.9729          0.9040          0.9163          0.2250
# csiq        0.045           0.9672          0.9772          0.9828          0.0493
# tid2013 1       0.527           0.7984          0.9298          0.9441          0.4562
# LIVEMD       3       5.972           0.9122         0.9684        0.9578         5.9157


######NEW：
# LIVE
# Testing  SRCC 0.9424,    PLCC 0.9502,    rmse 7.2852


# csiq
# [[0.95428571 0.94857143 0.86285714 0.96       0.85714286 0.91428571
#   0.94857143 0.92571429 0.87428571 0.96      ]]
# average MOSsrcc:0.9206
# [[0.94251861 0.93863724 0.85536832 0.93438316 0.90983925 0.91072467
#   0.94236812 0.87216889 0.8770269  0.92817296]]
# average MOSplcc:0.9111
# [[0.10552836 0.09327818 0.13498055 0.08966783 0.11712253 0.10269439
#   0.08324265 0.11022249 0.10531498 0.12573534]]
# average MOSrmse:0.1068


# tid2013

# Testing  SRCC 0.7925,    PLCC 0.8173,    rmse 0.9294

# Best test SRCC 0.827940, PLCC 0.858928, rmse 0.957641

# Best test SRCC 0.813374, PLCC 0.841819, rmse 1.038645
#  0.8324          0.8678          0.8153
# 0.8052          0.8323          0.8773
# 0.8326          0.8636          0.7227

# 0.824019
# 0.8546745
# 0.855714333

# LIVEMD
# Testing  SRCC 0.9500,    PLCC 0.9116,    rmse 11.7336

# CLIVE
# [[0.84530777 0.         0.         0.         0.         0.
#   0.         0.         0.         0.        ]]
# average MOSsrcc:0.0845
# [[0.8589332 0.        0.        0.        0.        0.        0.
#   0.        0.        0.       ]]
# average MOSplcc:0.0859
# [[10.75888176  0.          0.          0.          0.          0.
#    0.          0.          0.          0.        ]]
# average MOSrmse:1.0759