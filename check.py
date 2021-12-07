# import some common detectron2 utilities
import os

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# get image
from matplotlib import pyplot as plt

im = cv2.imread("/home/jang/Disk_500GB/Projects/Task/DrawingProject/static/images/c_0530.jpeg")

# Create config
cfg = get_cfg()
cfg.merge_from_file("/home/jang/anaconda3/envs/drawing_env/lib/python3.7/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = os.path.join("./model_detection_cat/model_final.pth")

# Create predictor
predictor = DefaultPredictor(cfg)

# Make prediction
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('r', v.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
