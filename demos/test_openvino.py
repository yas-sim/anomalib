# test1
import os, sys
import glob
import time

import cv2
import numpy as np

from anomalib.deploy import OpenVINOInferencer
from anomalib.post_processing import Visualizer

dataset_dir = './datasets/MVTec/bottle/test/broken_large'

images = glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True)
image = images[0]
img = cv2.imread(image)
print(img.shape)
#img = cv2.resize(img, (256, 256))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

model = './results/padim/mvtec/bottle/openvino/model.bin'
meta_data = './results/padim/mvtec/bottle/openvino/meta_data.json'
config = './demos/padim/config.yaml'
inferencer = OpenVINOInferencer(config=config, path=model, meta_data_path=meta_data)

task = ['classification', 'segmentation'][1]

niter = 100
stime = time.time()
inftime = 0
for i in range(niter):
    ss = time.time()
    predictions = inferencer.predict(image=img)
    ee = time.time()
    inftime += (ee-ss)
    # dir(predictions) : 'anomaly_map(900,900)', 'gt_mask', 'heat_map(900,900,3)', 
    # 'image', 'pred_label', 'pred_mask(900,900)', 'pred_score', 'segmentations(900,900,3)'

    hm = (np.expand_dims(predictions.anomaly_map, 2) * 255).astype(np.uint8)
    hm = cv2.resize(hm, (256, 256))
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

    im = cv2.resize(img, (256, 256))
    im = cv2.addWeighted(hm, 0.7, im, 0.3, 0)

    pm = (np.expand_dims(predictions.pred_mask, 2) * 255).astype(np.uint8)
    pm = cv2.resize(pm, (256, 256))
    ct, hc = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, ct, -1, color=(0,0,255), thickness=2)

    score = predictions.pred_score
    col = (0,0,255) if score > 0.7 else (0,255,0)
    score_text = '{:4.3f}'.format(score)
    cv2.rectangle(im, (0,0), (im.shape[0]-1, im.shape[1]-1), col, 6)
    cv2.putText(im, score_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 4)
    cv2.putText(im, score_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, col, 2)
etime = time.time()
print((etime-stime)/niter)
print(inftime / niter)

cv2.imshow('result', im)
cv2.waitKey(0)
