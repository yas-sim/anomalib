import os
import glob

import cv2

from anomalib.deploy import OpenVINOInferencer
from anomalib.post_processing import Visualizer

model = './results/padim/mvtec/bottle/openvino/model.bin'
meta_data = './results/padim/mvtec/bottle/openvino/meta_data.json'
config = './demos/padim/config.yaml'
mode = ['simple', 'full'][0]
task = ['classification', 'segmentation'][1]

inferencer = OpenVINOInferencer(config=config, path=model, meta_data_path=meta_data)
visualizer = Visualizer(mode=mode, task=task)

dataset_dir = './datasets/MVTec/bottle/test/broken_large'
images = glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True)
image = images[0]
img = cv2.imread(image)
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

predictions = inferencer.predict(image=img)
img = visualizer.visualize_image(predictions)
score = 'SCORE : {:4.2f}'.format(predictions.pred_score)
cv2.imshow(score, img)

cv2.waitKey(0)
