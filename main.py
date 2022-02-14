import os
import random

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data.detection_utils import build_transform_gen
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# TREE
def register_tree():
    """
    Register Dataset for tree detection
    :return: tree_metadata
    """
    register_coco_instances("train_dataset", {},
                            "/home/jang/Disk_1TB/Dataset/Drawing/tree/detection/train/output.json",
                            "/home/jang/Disk_1TB/Dataset/Drawing/tree/detection/train")
    register_coco_instances("val_dataset", {},
                            "/home/jang/Disk_1TB/Dataset/Drawing/tree/detection/val/output.json",
                            "/home/jang/Disk_1TB/Dataset/Drawing/tree/detection/val")

    tree_metadata = MetadataCatalog.get("train_dataset")
    print(tree_metadata)

    return tree_metadata


# Show tree images
def check_tree():
    dataset_dicts = DatasetCatalog.get("train_dataset")

    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=tree_metadata, scale=0.3)
        vis = visualizer.draw_dataset_dict(d)
        print(d["file_name"])
        testim = vis.get_image()[:, :, ::-1]
        plt.imshow(testim, interpolation='nearest')
        plt.show()


# Set configuration and train detection model
def train_tree_detection():
    """
    Detectron2 provides a key-value based config system that can be used to obtain standard, common behaviors.
    https://detectron2.readthedocs.io/en/latest/tutorials/configs.html
    1. Set config: pretrained model, iteration, batch size, class number, etc...
    2. Get detectron2 trainer initialized from a yacs config and start train
    3. Save model and return
    :return: cfg, trainer
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/jang/anaconda3/envs/drawing_env/lib/python3.7/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "ckpt/model_final_f6e8b1.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = 1000
    cfg.OUTPUT_DIR = "./model_detection_tree/"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (branches, trunk, roots)
    a = build_transform_gen(cfg, is_train=True)
    print(a)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # SAVING MODEL
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_dir="model_detection_tree")
    checkpointer.save("model_tree")

    return cfg, trainer


# Train classification model
def train_tree_classification():
    """
    1. Get cropped image data for classification
    2. For every class, train classification model
    :return: None
    """
    # TREE CLASSIFICATION
    print("Train Classification!")
    # A1 CROWN SHAPE
    classes = ["crown_arcade", "crown_ball", "branches"]
    crown_shape_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/A1_crown_shape", 3,
                                           classes,
                                           16, 20)
    crown_shape_model.save("model_classification_tree/crown_shape_model.h5")

    # A2 CROWN SHADE
    classes = ["shade", "no_shade"]
    crown_shade_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/A2_crown_shade", 2,
                                           classes,
                                           32, 20)
    crown_shade_model.save("model_classification_tree/crown_shade_model.h5")

    # B1 TRUNK SHAPE
    classes = ["trunk_base", "trunk_straight"]
    trunk_shape_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/B1_trunk_shape", 2,
                                           classes,
                                           32, 20)
    trunk_shape_model.save("model_classification_tree/trunk_shape_model.h5")

    # B2 TRUNK WAVE
    classes = ["trunk_wave", "no_wave"]
    trunk_wave_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/B2_trunk_wave", 2,
                                          classes,
                                          32, 20)
    trunk_wave_model.save("model_classification_tree/trunk_wave_model.h5")

    # B3 TRUNK LINES
    classes = ["lines", "no_lines"]
    trunk_lines_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/B3_trunk_lines", 2,
                                           classes,
                                           32, 20)
    trunk_lines_model.save("model_classification_tree/trunk_lines_model.h5")

    # B4 TRUNK SHADE
    classes = ["full_shade", "right_shade", "left_shade", "no_shade"]
    trunk_shade_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/B4_trunk_shade", 4,
                                           classes,
                                           8, 20)
    trunk_shade_model.save("model_classification_tree/trunk_shade_model.h5")

    # B5 TRUNK TILT
    classes = ["right", "left", "no_tilt"]
    trunk_tilt_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/B5_trunk_tilt", 3,
                                          classes,
                                          32, 20)
    trunk_tilt_model.save("model_classification_tree/trunk_tilt_model.h5")

    # B6 TRUNK PATTERN
    classes = ["trunk_round", "trunk_scratch", "no_pattern"]
    trunk_pattern_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/B6_trunk_pattern",
                                             3,
                                             classes, 32, 20)
    trunk_pattern_model.save("model_classification_tree/trunk_pattern_model.h5")

    # B7 LOW BRANCH
    classes = ["low_branch", "no_branch"]
    low_branch_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/B7_low_branch", 2,
                                          classes,
                                          16, 20)
    low_branch_model.save("model_classification_tree/low_branch_model.h5")

    # C1 FRUITS
    classes = ["fix", "no_fruit"]
    fruit_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/C1_fruit", 2, classes, 8,
                                     20)
    fruit_model.save("model_classification_tree/fruit_model.h5")

    # C2 CUT BRANCH
    classes = ["cut_branch", "no_cut_branch"]
    cut_branch_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/tree/classification/C2_cut_branch", 2,
                                          classes, 8,
                                          20)
    cut_branch_model.save("model_classification_tree/cut_branch_model.h5")


# Test tree models
def test_tree(cfg, trainer, tree_metadata):
    """
    Test model with validation dataset
    :param cfg: configs from train_tree_detection()
    :param trainer: trainer from train_tree_detection()
    :param tree_metadata: metadata from register_tree()
    :return: original image, output image
    """
    evaluator = COCOEvaluator("val_dataset", cfg, False, output_dir="model_detection_tree")

    val_loader = build_detection_test_loader(cfg, "val_dataset")
    result = inference_on_dataset(trainer.model, val_loader, evaluator)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_tree.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    cfg.DATASETS.TEST = ("val_dataset",)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get("val_dataset")

    for d in dataset_dicts:
        print(d)
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        print(outputs)
        v = Visualizer(im[:, :, ::-1],
                       metadata=tree_metadata,
                       scale=0.3,
                       )
        v1 = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.imshow(v1.get_image()[:, :, ::-1], interpolation='nearest')
        plt.show()
        cv2.imshow('result', v1.get_image()[:, :, ::-1])
        cv2.waitKey(0)

        return im, outputs


# CAT
def register_cat():
    """
    Register Dataset for cat detection
    :return: cat_metadata
    """
    register_coco_instances("train_dataset2", {},
                            "/home/jang/Disk_1TB/Dataset/Drawing/cat/detection/train/output.json",
                            "/home/jang/Disk_1TB/Dataset/Drawing/cat/detection/train")
    register_coco_instances("val_dataset2", {},
                            "/home/jang/Disk_1TB/Dataset/Drawing/cat/detection/val/output.json",
                            "/home/jang/Disk_1TB/Dataset/Drawing/cat/detection/val")
    cat_metadata = MetadataCatalog.get("train_dataset2")
    print(cat_metadata)

    return cat_metadata


# Show cat images
def check_cat():
    """
    Register Dataset for cat detection
    :return: cat_metadata
    """
    dataset_dicts = DatasetCatalog.get("train_dataset2")

    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=cat_metadata, scale=0.3)
        vis = visualizer.draw_dataset_dict(d)
        print(d["file_name"])
        testim = vis.get_image()[:, :, ::-1]
        plt.imshow(testim, interpolation='nearest')
        plt.show()


# Set configuration and cat detection model
def train_cat_detection():
    """
    Detectron2 provides a key-value based config system that can be used to obtain standard, common behaviors.
    https://detectron2.readthedocs.io/en/latest/tutorials/configs.html
    1. Set config: pretrained model, iteration, batch size, class number, etc...
    2. Get detectron2 trainer initialized from a yacs config and start train
    3. Save model and return
    :return: cfg, trainer
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/jang/anaconda3/envs/drawing_env/lib/python3.7/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("train_dataset2",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "ckpt/model_final_f6e8b1.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = 1000
    cfg.OUTPUT_DIR = "./model_detection_cat/"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (cat, head, body)
    a = build_transform_gen(cfg, is_train=True)
    print(a)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    # saving
    torch.save(trainer.model, "model_cat.pth")

    ### SAVING MODEL TRIAL2
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_dir="model_detection_cat")
    checkpointer.save("model_cat")

    return cfg, trainer


# Train classification model
def train_cat_classification():
    """
    1. Get cropped image data for classification
    2. For every class, train classification model
    :return: None
    """
    # Movement
    classes = ["dynamic", "static"]
    movement_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/cat/classification/movement", 2, classes,
                                        8, 20)
    movement_model.save("model_classification_cat/movement_model.h5")

    # Concept
    classes = ["conceptual", "no_conceptual"]
    concept_model = run_classification("/home/jang/Disk_1TB/Dataset/Drawing/cat/classification/concept", 2, classes, 8,
                                       20)
    concept_model.save("model_classification_cat/concept_model.h5")


# Test cat models
def test_cat(cfg, trainer, cat_metadata):
    """
    Test model with validation dataset
    :param cfg: configs from train_cat_detection()
    :param trainer: trainer from train_cat_detection()
    :param cat_metadata: metadata from register_cat()
    :return: original image, output image
    """
    evaluator = COCOEvaluator("val_dataset2", cfg, False, output_dir="model_detection_cat/")

    val_loader = build_detection_test_loader(cfg, "val_dataset2")
    result = inference_on_dataset(trainer.model, val_loader, evaluator)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_cat.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    cfg.DATASETS.TEST = ("val_dataset2",)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get("val_dataset2")

    for d in dataset_dicts:
        print(d)
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        print(outputs)
        v = Visualizer(im[:, :, ::-1],
                       metadata=cat_metadata,
                       scale=0.3,
                       )
        v1 = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.imshow(v1.get_image()[:, :, ::-1], interpolation='nearest')
        plt.show()
        cv2.imshow('result', v1.get_image()[:, :, ::-1])
        cv2.waitKey(0)

        return im, outputs


# Classification
def crop_classification(im, outputs):
    """
    Crop detected region
    :param im: original image
    :param outputs: image with detected regions
    :return: None
    """
    # CROPPING DETECTED BOXES
    boxes = {}
    array = []
    # label name
    label_array = []
    for label in outputs["instances"].to("cpu").pred_classes:
        if label.item() == 0:
            label = "branches"
        elif label.item() == 1:
            label = "roots"
        elif label.item() == 2:
            label = "trunk"

        label_array.append(label)

    # coordinate
    for coordinates in outputs["instances"].to("cpu").pred_boxes:
        coordinates_array = []
        for k in coordinates:
            coordinates_array.append(int(k))
        array.append(coordinates_array)

    # label name + coordinates
    for i in range(len(label_array)):
        boxes[label_array[i]] = array[i]

    print(boxes)
    img_array = []
    for k, v in boxes.items():
        print(k, ":", v)
        crop_img = im[v[1]:v[3], v[0]:v[2], :]
        plt.imshow(crop_img, interpolation='nearest')
        plt.show()
        cv2.imwrite(k + '.jpg', crop_img)
        img_array.append(crop_img)
        print(type(crop_img))


def run_classification(data_path, n_classes, classes, batch_size, epoch_number):
    """
    train classification
    :param data_path: data path
    :param n_classes: number of classes
    :param classes: class name list
    :param batch_size: batch size
    :param epoch_number: number of epochs
    :return:
    """
    train_data = ImageDataGenerator(rescale=1 / 255)
    train_generator = train_data.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')

    run_classification.model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')])

    run_classification.model.compile(loss='categorical_crossentropy',
                                     optimizer=RMSprop(lr=0.001), metrics=['acc'])

    total_sample = train_generator.n

    history = run_classification.model.fit_generator(
        train_generator,
        steps_per_epoch=int(total_sample / batch_size),
        epochs=epoch_number,
        verbose=1)

    return run_classification.model


def run_classification_pretrained(data_path, n_classes, classes, batch_size, epoch_number):
    train_data = ImageDataGenerator(rescale=1 / 255)
    train_generator = train_data.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    inputs = base_model.input
    x = base_model(inputs)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(n_classes)(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

    total_sample = train_generator.n

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=int(total_sample / batch_size),
        epochs=epoch_number,
        verbose=1)

    return model


if __name__ == '__main__':
    print("Hello world")

    test_type = 2

    if test_type == 1:
        tree_metadata = register_tree()
        # check_tree()
        cfg, trainer = train_tree_detection()
        train_tree_classification()
        im, outputs = test_tree(cfg, trainer, tree_metadata)
        crop_classification(im, outputs)

    elif test_type == 2:
        cat_metadata = register_cat()
        # check_cat()
        cfg, trainer = train_cat_detection()
        train_cat_classification()
        im, outputs = test_cat(cfg, trainer, cat_metadata)
        crop_classification(im, outputs)
