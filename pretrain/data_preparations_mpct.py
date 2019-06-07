import argparse
import os
import glob
from tqdm import tqdm

import pandas as pd
import cv2


def data_preparation(args):
    video_list = set()
    base_data_dir = args.base_dir
    img_size = args.resize
    img_dir = os.path.join(base_data_dir, 'image')
    label_dir = os.path.join(base_data_dir, 'bbox')
    data_labels = glob.glob(os.path.join(label_dir, 'catheter_*.csv'))
    column = ['index', 'c_x', 'c_y', 'w', 'h']
    dataset_dir = 'datasets/MPCT'

    for label in tqdm(data_labels):
        video_name = os.path.basename(label).split('catheter_')[1].split('.')[0]
        video_dir = os.path.join(dataset_dir, video_name)
        img_save_dir = os.path.join(video_dir, 'img')
        os.makedirs(img_save_dir, exist_ok=True)
        labels = pd.read_csv(label)
        bbox_annotation = open(os.path.join(video_dir, 'groundtruth_rect.txt'), 'w')
        for i in range(len(labels)):
            idx = labels['index'][i]
            img = cv2.imread(os.path.join(img_dir, video_name + f'_{str(idx)}.jpg'))
            if img is not None:
                video_list.add(video_name)
                img_y, img_x, _ = img.shape
                img = cv2.resize(img, (img_size, img_size))
                x = labels['c_x'][i] * img_size / img_x
                y = labels['c_y'][i] * img_size / img_y
                w = labels['w'][i] * img_size / img_x
                h = labels['h'][i] * img_size / img_y
                row = f'{str(int(x - w / 2))},{str(int(y - h / 2))},{str(int(w))},{str(int(h))}\n'
                img_name = '{:04d}.jpg'.format(i)
                cv2.imwrite(os.path.join(img_save_dir, img_name), img)
                bbox_annotation.write(row)
        bbox_annotation.close()
    video_list = list(video_list)
    with open('datasets/list/mpct.txt', 'w') as f:
        for item in video_list:
            f.write(item + '\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir')
    parser.add_argument('--resize', type=int, default=512)
    args = parser.parse_args()
    data_preparation(args)