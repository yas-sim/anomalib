import os, sys
import glob
import time
import random
import shutil

import cv2
import numpy as np

import yaml

# for openvino
from anomalib.deploy import OpenVINOInferencer
from anomalib.post_processing import Visualizer

# for pytorch lightning
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

base_dir = '.'
img_background = 'demos/background.png'

category = ''    # obtain from the config file later
dataset_dir = '' # obtain from the config file later

exit_flag = False
hm_blend = False    # Blend heat map on to the input image
pause_on_fault = False
canvas = None

threshold = 0.5     # Threshold value for predict_score
hm_weight = 0.8     # Heatmap and input image blending ratio

class Dataset:
    def __init__(self):
        self.bad_image_dirs = []
        self.good_image_dirs = []
        self.find_classes(os.path.join(base_dir, f'datasets/MVTec/{category}/test'))
        print(self.bad_image_dirs)
        print(self.good_image_dirs)
        self.bad_images = []
        self.good_images = []
        self.yield_rate = 0.9

    def find_classes(self, dataset_dir:str):
        dirs = glob.glob(os.path.join(dataset_dir, '*'))
        for dir in dirs:
            if os.path.isdir(dir):
                _, base = os.path.split(dir)
                if(base != 'good'):
                    self.bad_image_dirs.append(base)
                else:
                    self.good_image_dirs.append(base)

    def load(self, dataset_dir):
        for image_dir in self.bad_image_dirs:
            self.bad_images.append(glob.glob(os.path.join(dataset_dir, image_dir, '*.png'), recursive=True))
        for image_dir in self.good_image_dirs:
            self.good_images.append(glob.glob(os.path.join(dataset_dir, image_dir, '*.png'), recursive=True))

    def set_yield_rate(self, rate:float):
        rate = max(min(rate, 1.0), 0)
        self.yield_rate = rate

    def total_len(self, array_of_array:list):
        ttl = 0
        for tmparr in array_of_array:
            ttl += len(tmparr)
        return ttl

    def get_item(self, array_of_array:list, index:int):
        if self.total_len(array_of_array) <= index:
            return None
        for array in array_of_array:
            if index < len(array):
                return array[index]
            index -= len(array)
        return None

    def get_file_name(self):
        while True:
            rnd = random.random()
            if rnd > self.yield_rate:
                idx = int(self.total_len(self.bad_images) * random.random())
                file_name = self.get_item(self.bad_images, idx)
                return file_name
            else:
                idx = int(self.total_len(self.good_images) * random.random())
                file_name = self.get_item(self.good_images, idx)
                return file_name

class Average:
    def __init__(self):
        self.item = []
        self.num = 0
        self.pos = 0
    
    def set_num(self, val:int):
        self.num = val
        self.pos = 0
        self.fill(0)
    
    def fill(self, val):
        self.item = [ val for _ in range(self.num) ]

    def data(self, val):
        self.item[self.pos] = val
        self.pos = (self.pos + 1) % self.num
    
    def average(self):
        return sum(self.item) / len(self.item)


def draw_bar_graph(fps:float, ypos:int, ywid:int=40, maxfps:float=30):
    global canvas
    x0 = (1920 // 16) * 4
    y0 = ypos
    x1 = (1920 // 16) * 15
    y1 = y0 + ywid
    xval = int((x1-x0)*(fps/maxfps)+x0)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0,0,0), -1)
    cv2.rectangle(canvas, (x0, y0), (xval, y1), (255,128,0), -1)
    text = f'{fps:4.2f}'
    cv2.putText(canvas, text, ((1920 // 16) * 4, y0 + 36), cv2.FONT_HERSHEY_PLAIN, 3, (64,0,0), 3)


avg_lightning = Average()
avg_lightning.set_num(5)
avg_openvino = Average()
avg_openvino.set_num(5)

def infer_lightning(file_names:list, model, trainer, config, transform_config):
    global canvas, avg_lightning, exit_flag
    global hm_blend, pause_on_fault

    inftime = 0
    x0 = ((1920//2)-512)//2
    y0 = 200
    x1 = x0 + 512
    y1 = y0 + 512
    for file_name in file_names:
        dataset = InferenceDataset(file_name, image_size=tuple(config.dataset.image_size), transform_config=transform_config)
        dataloader = DataLoader(dataset)
        sinf = time.time()
        pred = trainer.predict(model=model, dataloaders=[dataloader])
        # dict_keys(['image(1,3,256,256)', 'image_path', 'anomaly_maps(1,1,256,256)', 'pred_scores', 'pred_labels[1]', 'pred_masks(1,1,256,256)'])
        einf = time.time()
        inftime = einf - sinf

        # Post Process ------------------------------------------------------------------------
        # heat map
        hm = pred[0]['anomaly_maps']
        hm = (hm[0,:,:,:].permute(1,2,0).to('cpu').detach().numpy() * 255).astype(np.uint8)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        # original image + heat map
        im = pred[0]['image']
        im = (im[0,:,:,:].permute(1,2,0).to('cpu').detach().numpy() * 255).astype(np.uint8).copy()
        if hm_blend:
            im = cv2.addWeighted(hm, hm_weight, im, 1-hm_weight, 0)

        # draw prediction score and frame. embed the image into background image
        score = pred[0]['pred_scores'][0]
        fault = False
        if score >= threshold:
            fault = True
            col = (0, 0, 255)
            # find contours of defects and draw contours
            pm = pred[0]['pred_masks']
            pm = (pm[0,:,:,:].permute(1,2,0).to('cpu').detach().numpy() * 255).astype(np.uint8)
            ct, hc = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, ct, -1, color=(0,0,255), thickness=2)
        else:
            col = (0,255,0)
        score_text = f'{score:4.3f}'
        cv2.rectangle(im, (0,0), (im.shape[0]-1, im.shape[1]-1), col, 6)
        cv2.putText(im, score_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 4)
        cv2.putText(im, score_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, col, 2)
        im = cv2.resize(im, (512, 512))
        canvas[y0:y1, x0:x1, :] = im

        # draw bar graph
        avg_lightning.data(1/inftime)
        draw_bar_graph(avg_lightning.average(), (1080//16)*12)

        cv2.imshow('screen', canvas)
        key = cv2.waitKey(0 if pause_on_fault and fault else 1)
        if key == 27 or key == ord('q'):
            exit_flag = True
            return
        elif key == ord('p'):
            pause_on_fault = False if pause_on_fault else True
            cv2.rectangle(canvas, (0,0), (400, 40), (255,255,255), -1)
            if pause_on_fault:
                cv2.putText(canvas, 'PAUSE ON FAULT', (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,128,0), 2)
        elif key == ord('h'):
            hm_blend = False if hm_blend else True


def infer_openvino(file_names:list, inferencer):
    global canvas, avg_openvino, exit_flag
    global hm_blend, pause_on_fault

    x0 = ((1920//2)-512)//2 + 1920 // 2
    y0 = 200
    x1 = x0 + 512
    y1 = y0 + 512
    inftime = 0
    for file_name in file_names:
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sinf = time.time()
        predictions = inferencer.predict(image=img)
        einf = time.time()
        inftime = einf - sinf
        # dir(predictions) : 'anomaly_map(900,900)', 'gt_mask', 'heat_map(900,900,3)', 
        # 'image', 'pred_label', 'pred_mask(900,900)', 'pred_score', 'segmentations(900,900,3)'

        # Post Process ------------------------------------------------------------------------
        # heat map
        hm = (np.expand_dims(predictions.anomaly_map, 2) * 255).astype(np.uint8)
        hm = cv2.resize(hm, (256, 256))
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        # original image + heat map
        im = cv2.resize(img, (256, 256))
        if hm_blend:
            im = cv2.addWeighted(hm, hm_weight, im, 1-hm_weight, 0)

        # draw prediction score and frame. embed the image into background image
        score = predictions.pred_score
        fault = False
        if score >= threshold:
            fault = True
            col = (0,0,255)
            # find contours of defects and draw contours
            pm = (np.expand_dims(predictions.pred_mask, 2) * 255).astype(np.uint8)
            pm = cv2.resize(pm, (256, 256))
            ct, hc = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, ct, -1, color=(0,0,255), thickness=2)
        else:
            col = (0, 255, 0)
        score_text = f'{score:4.3f}'
        cv2.rectangle(im, (0,0), (im.shape[0]-1, im.shape[1]-1), col, 6)
        cv2.putText(im, score_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 4)
        cv2.putText(im, score_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, col, 2)
        im = cv2.resize(im, (512, 512))
        canvas[y0:y1, x0:x1, :] = im

        # draw bar graph
        avg_openvino.data(1/inftime)
        draw_bar_graph(avg_openvino.average(), (1080//16)*14)

        cv2.imshow('screen', canvas)
        key = cv2.waitKey(0 if pause_on_fault and fault else 1)
        if key == 27 or key == ord('q'):
            exit_flag = True
            return
        elif key == ord('p'):
            pause_on_fault = False if pause_on_fault else True
            cv2.rectangle(canvas, (0,0), (400, 40), (255,255,255), -1)
            if pause_on_fault:
                cv2.putText(canvas, 'PAUSE ON FAULT', (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,128,0), 2)
        elif key == ord('h'):
            hm_blend = False if hm_blend else True


def main():
    global canvas
    global category, dataset_dir

    config = os.path.join(base_dir, 'demos/padim/config.yaml')
    # read YAML config file
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # extract some parameters from the config file
    category = cfg['dataset']['category']
    dataset_dir = cfg['dataset']['path']
    dataset_dir = os.path.join(base_dir, dataset_dir, category, 'test')
    print('[ INFO ] Inspect category =', category)
    print('[ INFO ] Dataset dir =', dataset_dir)

    # Load dataset (path search)
    dataset = Dataset()
    dataset.load(dataset_dir)
    dataset.set_yield_rate(0.7)

    # Initialize Pytorch Lightning
    ptl_weights = os.path.join(base_dir, f'results/padim/mvtec/{category}/weights/model.ckpt')
    ptl_config = get_configurable_parameters(config_path=config)
    ptl_config.trainer.resume_from_checkpoint = str(ptl_weights)
    ptl_config.visualization.show_images = False
    ptl_config.visualization.mode = ['simple', 'full'][0]
    ptl_config.visualization.save_images = False
    ptl_model = get_model(ptl_config)
    callbacks = get_callbacks(ptl_config)
    ptl_trainer = Trainer(callbacks=callbacks, **ptl_config.trainer)
    ptl_transform_config = ptl_config.dataset.transform_config.val if "transform_config" in ptl_config.dataset.keys() else None

    # Initialize OpenVINO
    ov_model = os.path.join(base_dir, f'results/padim/mvtec/{category}/openvino/model.bin')
    meta_data = os.path.join(base_dir, f'results/padim/mvtec/{category}/openvino/meta_data.json')
    ov_inferencer = OpenVINOInferencer(config=config, path=ov_model, meta_data_path=meta_data)

    # Grab full screen
    canvas = cv2.imread(os.path.join(base_dir, img_background))
    cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('screen', canvas)

    while True:
        file_names = []
        for i in range(20):
            file_names.append(dataset.get_file_name())

        infer_lightning(file_names, ptl_model, ptl_trainer, ptl_config, ptl_transform_config)
        if exit_flag == True:
            return
        shutil.rmtree(os.path.join(base_dir, f'results/padim/mvtec/{category}/lightning_logs/')) # Delete unnecessary log files

        infer_openvino(file_names, ov_inferencer)
        if exit_flag == True:
            return

if __name__ == '__main__':
    sys.exit(main())
