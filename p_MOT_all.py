# encoding: utf-8
"""
@author:  jixunbo
@contact: jixunbo@gmail.com
"""


import os
import os.path as osp
import csv
from shutil import copyfile
import cv2
import time
import configparser
import numpy as np
import random

# You only need to change these lines to your dataset download path
download_path = '/Users/jixunbo/Desktop/ReIDataset/MOT17Det'
label_path = '/Users/jixunbo/Desktop/ReIDataset/MOT17Labels'
save_path = '/Users/jixunbo/Desktop/MOT17ReID'
crop_W = 64
crop_H = 128
val_num = 2  # To random select number of frames allcated to validation
query_num = 3  # To random select number of frames allcated to query
min_frames = 5  # To check the sequence longer than 5 frames


def _sequence(seq_name, _mot_dir, _mot17_label_dir, _dets, _vis_threshold):
    _train_folders = os.listdir(os.path.join(_mot_dir, 'train'))
    _test_folders = os.listdir(os.path.join(_mot_dir, 'test'))
    if seq_name:
        assert seq_name in _train_folders or seq_name in _test_folders, \
            'Image set does not exist: {}'.format(seq_name)

    if seq_name in _train_folders:
        seq_path = osp.join(_mot_dir, 'train', seq_name)
#         label_path = osp.join(self._label_dir, 'train', 'MOT16-'+seq_name[-2:])
        mot17_label_path = osp.join(_mot17_label_dir, 'train')
    else:
        seq_path = osp.join(_mot_dir, 'test', seq_name)
#         label_path = osp.join(self._label_dir, 'test', 'MOT16-'+seq_name[-2:])
        mot17_label_path = osp.join(_mot17_label_dir, 'test')
#     raw_label_path = osp.join(self._raw_label_dir, 'MOT16-'+seq_name[-2:])

    config_file = osp.join(seq_path, 'seqinfo.ini')

    assert osp.exists(config_file), \
        'Config file does not exist: {}'.format(config_file)

    config = configparser.ConfigParser()
    config.read(config_file)
    seqLength = int(config['Sequence']['seqLength'])
    imDir = config['Sequence']['imDir']

    imDir = osp.join(seq_path, imDir)
    gt_file = osp.join(seq_path, 'gt', 'gt.txt')

    total = []
    train = []
    val = []

    visibility = {}
    boxes = {}
    dets = {}

    for i in range(1, seqLength + 1):
        boxes[i] = {}
        visibility[i] = {}
        dets[i] = []

    no_gt = False
    if osp.exists(gt_file):
        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                # class person, certainity 1, visibility >= 0.25
                if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= _vis_threshold:
                    # Make pixel indexes 0-based, should already be 0-based (or not)
                    x1 = int(row[2]) - 1
                    y1 = int(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + int(row[4]) - 1
                    y2 = y1 + int(row[5]) - 1
                    bb = np.array([x1, y1, x2, y2], dtype=np.float32)
                    boxes[int(row[0])][int(row[1])] = bb
                    visibility[int(row[0])][int(row[1])] = float(row[8])
    else:
        no_gt = True
    det_file = osp.join(
        mot17_label_path,
        f"{seq_name}-{_dets[:-2]}",
        'det',
        'det.txt')

    if osp.exists(det_file):
        with open(det_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                x1 = float(row[2]) - 1
                y1 = float(row[3]) - 1
                # This -1 accounts for the width (width of 1 x1=x2)
                x2 = x1 + float(row[4]) - 1
                y2 = y1 + float(row[5]) - 1
                score = float(row[6])
                bb = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                dets[int(row[0])].append(bb)

    for i in range(1, seqLength + 1):
        im_path = osp.join(imDir, "{:06d}.jpg".format(i))

        sample = {'seq_name': seq_name,
                  'gt': boxes[i],
                  'im_path': im_path,
                  'vis': visibility[i],
                  'dets': dets[i], }

        total.append(sample)

    return total, no_gt


def build_samples(data, _seq_name):
    """Builds the samples out of the sequence."""

    tracks = {}
    timepoint = time.time()
    for sample in data:
        #         print(sample)

        im_path = sample['im_path']
        gt = sample['gt']

        # 下面这一段的逻辑是：
        # sample是第一帧，开始sample的第一次循环，gt里包含第一帧的boundingbox之类的
        # 一开始tracks是没有值的，所以下面第一次kv in tracks没有作用，到了kv in gt
        # 的时候，tracks开始把第一帧中的人物加载进去，track[2]就表示id为2的人这一帧
        # 的信息，然后到第二个sample就是第二帧，这时kv in tracks就有用了，k=2时，如果
        # k在第二帧的gt里也出现了，则tracks里的v会append这一帧的gt，就有
        # track[2]=[{gt第一帧},{gt第二帧}]，原来tracks出现过的人物过完之后，del gt是
        # 删掉已经出现过的人的信息，然后到了下面的kv in gt 这时由于出现过的人都删掉了，所以
        # gt里只剩新人了，这时再把gt里这一帧的新人加到tracks里，以此类推一帧一帧把人都加进去

        for k, v in tracks.items():
            if k in gt.keys():
                v.append({'id': k, 'im_path': im_path, 'gt': gt[k]})
                del gt[k]
        # For all remaining BB in gt new tracks are created
        for k, v in gt.items():
            tracks[k] = [{'id': k, 'im_path': im_path, 'gt': v}]
    # sample max_per_person images and filter out tracks smaller than 4 samples
    # outdir = get_output_dir("siamese_test")
    res = []
    # print(len(tracks.items()))
    for k, v in tracks.items():

        l = len(v)
        pers = []

        for i in range(l):
            pers.append(build_crop(
                _seq_name, v[i]['im_path'], v[i]['gt'], crop_W, crop_H))

            # for i,v in enumerate(pers):
            #   cv2.imwrite(osp.join(outdir, str(k)+'_'+str(i)+'.png'),v)

        res.append(np.array(pers))
    print("[*] Loaded {} persons from {} sequence.".format(len(res), _seq_name))
    print("Time used: {} seconds".format(time.time() - timepoint))
    return res
    #########################################

#     r = []
#     for idx, pers in enumerate(res):
#         for im in pers:
#             im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#             # im = Image.fromarray(im)
#             r.append([im, idx])
#         # print(idx,'dfdfd')


#     return r
    # self.data  format is [[idx,im],[idx,im]....]

    #########################################
    # print(np.shape(self.data[0]))

def build_crop(_seq_name, im_path, gt, crop_W, crop_H):
    im = cv2.imread(im_path)
    height, width, channels = im.shape
    # blobs, im_scales = _get_blobs(im)
    # im = blobs['data'][0]
    # gt = gt * im_scales[0]
    # clip to image boundary
    w = gt[2] - gt[0]
    h = gt[3] - gt[1]
    context = 0
    gt[0] = np.clip(gt[0] - context * w, 0, width - 1)
    gt[1] = np.clip(gt[1] - context * h, 0, height - 1)
    gt[2] = np.clip(gt[2] + context * w, 0, width - 1)
    gt[3] = np.clip(gt[3] + context * h, 0, height - 1)

    im = im[int(gt[1]):int(gt[3]), int(gt[0]):int(gt[2])]

    im = cv2.resize(im, (int(crop_W),
                         int(crop_H)), interpolation=cv2.INTER_LINEAR)

    return {'seq_name': _seq_name, 'frame_id': im_path[-6:], 'img': im}




if not os.path.isdir(download_path):
    print('please change the download_path')
if not os.path.isdir(label_path):
    print('please change the label_path')

# save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

_train_folders = os.listdir(os.path.join(download_path, 'train'))
_test_folders = os.listdir(os.path.join(download_path, 'test'))
seq_list = ['MOT17-13', 'MOT17-09', 'MOT17-11',
            'MOT17-10', 'MOT17-04', 'MOT17-05', 'MOT17-02']

train_save_path = save_path + '/train'
val_save_path = save_path + '/val'
query_save_path = save_path + '/query'
gallery_save_path = save_path + '/gallery'

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)
    os.mkdir(query_save_path)
    os.mkdir(gallery_save_path)

label_generator = 1
for seq_name in seq_list:
    data, gt = _sequence(seq_name, download_path, label_path, 'FRCNN17', 0.25)
    pers_clusters = build_samples(data, seq_name)
    num_pers = len(pers_clusters)
    _split = np.random.permutation(num_pers)
    # test_split = _split[:int(num_pers / 5)]
    # trainval_split = _split[int(num_pers / 5):]
    trainval_split=_split
    test_split=[]
    # print(trainval_split.shape, '--', test_split.shape)

    for pers in trainval_split:
        # To check if each person has more than 4 image
        if len(pers_clusters[pers]) > min_frames:
            val_count = 0
            random.shuffle(pers_clusters[pers])
            for sample in pers_clusters[pers]:
                # print(len(pers_clusters[pers]))
                # if val_count < val_num:  # first 2 images is used as val image
                #     dst_path = osp.join(val_save_path, str(label_generator))
                # else:
                dst_path = train_save_path

                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                    # first image is used as val image
                cv2.imwrite(osp.join(dst_path, '{}_1_{}_{}'.format(label_generator,
                    sample['seq_name'], sample['frame_id'])), sample['img'])
                val_count += 1
            label_generator += 1

    for pers in test_split:

        if len(pers_clusters[pers]) > min_frames:
            query_count = 0
            random.shuffle(pers_clusters[pers])

            for sample in pers_clusters[pers]:

                if query_count < query_num:  # first 2 images is used as val image
                    dst_path = osp.join(query_save_path, str(label_generator))
                else:
                    dst_path = osp.join(gallery_save_path,
                                        str(label_generator))
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                cv2.imwrite(osp.join(dst_path, '{}_{}'.format(
                    sample['seq_name'], sample['frame_id'])), sample['img'])
                query_count += 1

            label_generator += 1
