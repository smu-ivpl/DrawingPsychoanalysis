import os
import json
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt

from tqdm import tqdm

import warnings

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

warnings.simplefilter(action='ignore', category=Warning)


class TextConfig(Config):
    """
    Configuration for training on the text dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "text"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class TextDataset(utils.Dataset):

    def load_text_data(self, dataset_dir, subset):
        """
        Load a subset of the text dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("text", 1, "text")

        # Train or validation dataset?
        assert subset in ["train", "validation", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "text",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "text":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool_), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "text":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train_detection(model, dataset, config, epochs=100):
    """Train the model."""
    # Training dataset.
    dataset_train = TextDataset()
    dataset_train.load_text_data(dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TextDataset()
    dataset_val.load_text_data(dataset, "validation")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train_detection(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=epochs,
                          layers='3+')


def evaluate_detection(model, dataset, config, eval_data):
    # Validation dataset
    dataset_eval = TextDataset()
    dataset_eval.load_text_data(dataset, eval_data)
    dataset_eval.prepare()

    # evaluation
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.

    # image_ids = np.random.choice(dataset_val.image_ids, 48)
    image_ids = []

    print('Data num: ' + str(len(dataset_eval.image_info)))
    for i in range(len(dataset_eval.image_info)):  # val 데이터셋 개수 넣어주기
        image_ids.append(i)
    APs = []
    for image_id in tqdm(image_ids):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_eval, config, image_id,
                                                                                  use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


# crop dot rois
def crop_bbox(image, bbox, padding):
    rois = []
    for b in bbox:
        # skimage crop -> image[x1:x2,y1:y2]
        # roi = image[b[0]:b[2], b[1]:b[3]]
        p = [b[0] - padding, b[2] + padding, b[1] - padding, b[3] + padding]  # padding
        if p[0] < 0: p[0] = 0
        if p[2] < 0: p[2] = 0
        if p[1] > image.shape[0]: p[1] = image.shape[0]
        if p[3] > image.shape[1]: p[3] = image.shape[1]

        roi = image[p[0]:p[1], p[2]:p[3]]

        plt.imshow(roi)
        plt.show()

        rois.append(roi)

    return rois


def detect_and_color_splash(model, padding, image_path,  image_name=None, img_file_name=None):
    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_name))
    # Read image
    image = skimage.io.imread(image_name)
    image = skimage.color.gray2rgb(image)

    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # bounding box visualize
    class_names = ['background', 'text']
    bbox = utils.extract_bboxes(r['masks'])
    file_name_bb = "bb_splash_{}".format(img_file_name)
    save_path_bb = os.path.join(image_path, 'result', file_name_bb)

    # save cropped dot image
    rois = crop_bbox(image, bbox, padding)
    i = 0
    # image_name = image_path.split('/')[-1][:-4]
    for roi in rois:
        file_name = 'crop_' + img_file_name
        skimage.io.imsave(
            image_path + "/crop_result/" + file_name.split('.')[0] + '_'+str(i) + '.jpg', roi)
        print("Saved to ", file_name.split('.')[0] + '_'+str(i) + '.jpg')
        i += 1

    visualize.display_instances(save_path_bb, image, bbox, r['masks'], r['class_ids'], class_names, r['scores'])

    # Color splash
    splash = color_splash(image, r['masks'])
    # Save output
    file_name = "splash_{}".format(img_file_name)
    save_path = os.path.join(image_path, 'result', file_name)
    skimage.io.imsave(save_path, splash)

    print("Saved to ", save_path)
    print('')


# Just crop annotated region
def just_crop(dataset, config, data_type):
    # Validation dataset
    dataset_crop = TextDataset()
    dataset_crop.load_text_data(dataset, data_type)
    dataset_crop.prepare()

    # image_ids = np.random.choice(dataset_val.image_ids, 48)
    image_ids = []

    print('Data num: ' + str(len(dataset_crop.image_info)))
    for i in range(len(dataset_crop.image_info)):  # 데이터셋 개수 넣어주기
        image_ids.append(i)

    for image_id in tqdm(image_ids):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_crop, config, image_id,
                                                                                  use_mini_mask=False)
        count = 1
        for b in gt_bbox:
            p = [b[0], b[2], b[1], b[3]]  # padding
            if p[0] < 0: p[0] = 0
            if p[2] < 0: p[2] = 0
            if p[1] > image.shape[0]: p[1] = image.shape[0]
            if p[3] > image.shape[1]: p[3] = image.shape[1]

            roi = image[p[0]:p[1], p[2]:p[3]]

            plt.imshow(roi)
            plt.show()

            # Save output
            file_name = dataset_crop.image_info[image_id]['id'].split('.')[0] + "_{}".format(count) + '.jpg'
            save_path = os.path.join('/home/jijang/projects/Task/OCR/text_dataset_cropped/', data_type, file_name)
            skimage.io.imsave(save_path, roi)

            count += 1

