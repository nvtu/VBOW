import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Process
import os
import subprocess


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))


def employ_task(params):
    for p in params:
        image_filepath, fout_path = p
        if os.path.exists(fout_path): continue
        print('Processing {}'.format(image_filepath))
        color_img = cv2.imread(image_filepath)
        gray_img = to_gray(color_img)
        kp, desc = gen_sift_features(gray_img)
        if isinstance(desc, type(None)): continue
        np.save(fout_path, desc / 255)


def multiprocess_extract(params):
    num_processes = 20
    task_per_process = len(params) // num_processes + 1
    tasks = []
    for i in range(num_processes):
        sub_p = params[i * task_per_process : min((i + 1) * task_per_process, len(params))]
        tasks.append(Process(target=employ_task, args=(sub_p,)))
        tasks[-1].start()
    for task in tasks:
        task.join()


if __name__ == '__main__':
    dataset_path = '/home/tuninh_2411/LSC_DATA'
    sift_feat_dir = str(Path.cwd() / 'sift_feat')
    if not os.path.isdir(sift_feat_dir):
        os.mkdir(sift_feat_dir)
    
    # Extract SIFT features for each image and save each of them separately
    images_info = []
    for root, dirs, files in os.walk(dataset_path):
        for d in dirs:
            if d == '.DS_Store': continue
            _d = os.path.join(sift_feat_dir, d)
            if not os.path.isdir(_d):
                os.mkdir(_d)
        for f in files:
            if f == '.DS_Store': continue
            file_name = f.split('.')[0]
            fout_path = os.path.join(root, '{}.npy'.format(file_name)).replace(dataset_path, sift_feat_dir)
            if os.path.exists(fout_path): continue
            _f = os.path.join(root, f)
            images_info.append((_f, fout_path))
#            color_img = cv2.imread(_f)
#            gray_img = to_gray(color_img)
#            kp, desc = gen_sift_features(gray_img)
#            if isinstance(desc, type(None)): continue
#            np.save(fout_path, desc / 255)
    multiprocess_extract(images_info)

    # Combine extracted SIFT features into one file
    print('Combine SIFT features')
    fout_path = 'combined_sift_feat.npy'
    if os.path.exists(fout_path): exit(0)
    combination = []
    num_feat = 0
    for root, dirs, files in os.walk(sift_feat_dir):
        for i, f in enumerate(files):
            _f = os.path.join(root, f)
            try:
                temp = np.load(_f)
                num_feat += temp.shape[0]
                print(num_feat)
                combination.append(temp)
            except Exception as e:
                print(e)
                cmd = 'rm {}'.format(_f)
                subprocess.call(cmd, shell=True)
    combination = np.vstack(np.array(combination))
    np.save(fout_path, combination)
