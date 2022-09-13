# test 2
import os, sys
import glob
import time

import cv2
import numpy as np

from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

dataset_dir = './datasets/MVTec/bottle/test/broken_large'

images = glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True)
image = images[0]
#img = cv2.imread(image)
#img = cv2.resize(img, (256, 256))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

config = './demos/padim/config.yaml'
ptl_weights = './results/padim/mvtec/bottle/weights/model.ckpt'
visualization_mode = ['simple', 'full'][1]

"""Run inference."""
config = get_configurable_parameters(config_path=config)
config.trainer.resume_from_checkpoint = str(ptl_weights)
config.visualization.show_images = False
config.visualization.mode = visualization_mode
config.visualization.save_images = False

model = get_model(config)
callbacks = get_callbacks(config)

trainer = Trainer(callbacks=callbacks, **config.trainer)

transform_config = config.dataset.transform_config.val if "transform_config" in config.dataset.keys() else None

niter = 1
stime = time.time()
inftime = 0
for i in range(niter):
    dataset = InferenceDataset(image, image_size=tuple(config.dataset.image_size), transform_config=transform_config)
    dataloader = DataLoader(dataset)
    ss = time.time()
    pred = trainer.predict(model=model, dataloaders=[dataloader])
    ee = time.time()
    inftime += (ee-ss)
    # dict_keys(['image(1,3,256,256)', 'image_path', 'anomaly_maps(1,1,256,256)', 'pred_scores', 'pred_labels[1]', 'pred_masks(1,1,256,256)'])

    hm = pred[0]['anomaly_maps']
    hm = (hm[0,:,:,:].permute(1,2,0).to('cpu').detach().numpy() * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

    pd = pred[0]['pred_masks']
    pd = (pd[0,:,:,:].permute(1,2,0).to('cpu').detach().numpy() * 255).astype(np.uint8)

    im = pred[0]['image']
    im = (im[0,:,:,:].permute(1,2,0).to('cpu').detach().numpy() * 255).astype(np.uint8)

    im = cv2.addWeighted(hm, 0.7, im, 0.3, 0)
    ct, hc = cv2.findContours(pd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, ct, -1, color=(0,0,255), thickness=2)

    score = pred[0]['pred_scores'][0]
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
