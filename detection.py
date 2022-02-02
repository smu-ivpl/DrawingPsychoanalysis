from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from tensorflow import keras
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import cv2
import random

matplotlib.use('agg')
count = 0


def init():
    cfg = get_cfg()

    import shutil
    env_path = shutil.which('python')
    cfg.merge_from_file(env_path.replace("/bin/python",
        "/lib/python3.7/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "model_pretrained/model_final_f6e8b1.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2

    # Learning Rate
    cfg.SOLVER.BASE_LR = 0.00025
    # Max Iteration
    cfg.SOLVER.MAX_ITER = 1000
    # Batch Size
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    cfg.DATALOADER.NUM_WORKERS = 2

    # initialize from model zoo
    # 3 classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    cfg.MODEL.WEIGHTS = os.path.join("./model_pretrained/model_detection_tree/model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model

    models = {}
    models['tree'] = {}
    models['cat'] = {}
    models['life'] = {}

    models['tree']['default'] = DefaultPredictor(cfg)

    models['tree']['crown_shape'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/crown_shape_model.h5")
    models['tree']['crown_shade'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/crown_shade_model.h5")
    models['tree']['crown_fruit'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/fruit_model.h5")
    models['tree']['cut_branch'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/cut_branch_model.h5")
    models['tree']['trunk_shape'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/trunk_shape_model.h5")
    models['tree']['trunk_wave'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/trunk_wave_model.h5")
    models['tree']['trunk_lines'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/trunk_lines_model.h5")
    models['tree']['trunk_shade'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/trunk_shade_model.h5")
    models['tree']['trunk_tilt'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/trunk_tilt_model.h5")
    models['tree']['trunk_pattern'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/trunk_pattern_model.h5")
    models['tree']['low_branch'] = keras.models.load_model(
        "model_pretrained/model_classification_tree/low_branch_model.h5")

    cfg.MODEL.WEIGHTS = os.path.join("./model_pretrained/model_detection_cat/model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

    models['cat']['default'] = DefaultPredictor(cfg)
    models['cat']['concept'] = keras.models.load_model("model_pretrained/model_classification_cat/concept_model.h5")
    models['cat']['movement'] = keras.models.load_model(
        "model_pretrained/model_classification_cat/movement_model.h5")

    # cfg.MODEL.WEIGHTS = "./model_pretrained/model_life_space/best_accuracy.pth"
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

    # models['life']['default'] = keras.models.load_model("model_pretrained/model_life_space/mask_rcnn_text_0001.h5")

    return models


# sentence forming
def adj_sentence(attr1, attr2, adj, adj3):
    sentences = ["%s이 %s인 것으로 보아 당신은 %s이고 %s이 있는 것으로 보여집니다.",
                 "%s이(가) %s로 나타남으로써 당신은 %s이고 %s 경향이 보여집니다.",
                 "당신의 그림의 %s이(가) %s인 것으로 보아 당신은 %s이고 %s이 있어보입니다.",
                 "당신이 그린 그림을 보니 %s이(가) %s인 것을 보아 %s이고 %s를 내제하고 있는 것이 보여집니다."]

    sentence = random.choice(sentences)

    sentence = sentence % (attr1, attr2, adj, adj3)
    return sentence


def getWhitePercent(img, total, x0, x1, y0, y1):
    global count
    white = 0
    other = 0

    for x in range(int(x0), int(x1)):
        for y in range(int(y0), int(y1)):
            # print("img[x][y]: ", img[x][y])
            if (img[x][y] >= 245).all():
                white += 1
            else:
                other += 1

    percentage = white * 100 / total
    # print(percentage)

    if percentage < 85:
        count += 1
    else:
        count += 0

    return count


# Cat colorfulness
def image_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)  # Red channel - Blue channel
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    std_root = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    mean_root = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return std_root + (0.3 * mean_root)


def test_life_space(models, img_dir):
    pass


def test_tree(models, img_dir):
    dest = img_dir.replace(os.path.basename(img_dir), '')
    path_details = {}

    models['default'].cfg.DATASETS.TEST = (img_dir,)

    test_metadata = MetadataCatalog.get(img_dir)

    # region Detection Part
    print("test tree!!!!!!!!")
    print("tree file: " + img_dir)
    filename = img_dir.split('/')[-1]

    im = cv2.imread(img_dir)
    outputs = models['default'](im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata,
                   scale=0.5
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Fix tree image size: 350x600
    resized_img = cv2.resize(out.get_image()[:, :, ::-1], dsize=(350, 600), interpolation=cv2.INTER_AREA)
    name = "{}_{}.jpg".format(filename.split('.')[0], "detected")
    path_detected = os.path.join(dest, name)
    cv2.imwrite(path_detected, resized_img)

    img1 = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)

    scale_percent = 50
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', img.shape)

    y1 = img.shape[1] / 5
    y2 = img.shape[1] - y1

    total = y1 * img.shape[0]
    count = getWhitePercent(img, total, 0, img.shape[0], y2, img.shape[1])

    if count >= 3:
        results = "꽉 찬 공간밀도, "
    else:
        results = "비어있는 공간밀도, "

    # final_sentence = sentence00 + "\r\n"
    final_sentence = results

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

    for coordinates in outputs["instances"].to("cpu").pred_boxes:
        coordinates_array = []
        for k in coordinates:
            coordinates_array.append(int(k))
        array.append(coordinates_array)

    for i in range(len(label_array)):
        boxes[label_array[i]] = array[i]

    print("STEP1 BOXES: ", boxes)

    # Crop Image
    img_array = []
    for k, v in boxes.items():
        crop_img = im[v[1]:v[3], v[0]:v[2], :]

        ratio = 350.0 / crop_img.shape[1]
        dim = (350, int(crop_img.shape[0] * ratio))

        detected_file = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
        name = "{}_{}.jpg".format(filename.split('.')[0], k)
        path_details[k] = os.path.join(dest, name)
        cv2.imwrite(path_details[k], detected_file)
        img_array.append(crop_img)

    if len(img_array) == 3:
        sentence0 = "뿌리 보임, "
        final_sentence = final_sentence + sentence0

    img = img_array[0]

    bottle_resized = resize(img, (224, 224))
    bottle_resized = np.expand_dims(bottle_resized, axis=0)
    # endregion

    # region Class: crown shape
    print("tree_model_crown_shape")
    pred = models['crown_shape'].predict(bottle_resized)

    crown_str = "왕관: "
    if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2]:
        crown_str += "아케이드 모양, "

    elif pred[0, 1] > pred[0, 2]:
        crown_str += "공 모양, "
    # endregion

    # region Class: crown shade
    print("tree_model_crown_shade")
    pred = models['crown_shade'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        crown_str += "그림자 진, "
    # endregion

    # region Class: crown fruit
    print("tree_model_crown_fruit")
    pred = models['crown_fruit'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        crown_str += "과일 달린, "
    # endregion

    if crown_str != "왕관: ":
        final_sentence += crown_str

    # region Class: cut branch
    print("tree_model_cut_branch")
    pred = models['cut_branch'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        final_sentence += "잘린 나뭇가지, "

    # Last, check tree attributes for trunk
    img2 = img_array[1]
    bottle_resized2 = resize(img2, (224, 224))
    bottle_resized2 = np.expand_dims(bottle_resized2, axis=0)
    # endregion

    # region Class: trunk shape
    print("tree_model_trunk_shape")
    pred = models['trunk_shape'].predict(bottle_resized2)

    trunk_str = "나무기둥: "
    if pred[0, 0] > pred[0, 1]:
        trunk_str += "양쪽으로 넓은, "
    else:
        trunk_str += "직선 모양의, "
    # endregion

    # region Class: trunk wave
    print("tree_model_trunk_wave")
    pred = models['trunk_wave'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1]:
        trunk_str += "구불거리는 형태의, "
    # endregion

    # region Class: trunk lines
    print("tree_model_trunk_lines")
    pred = models['trunk_lines'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1]:
        trunk_str += "흩어진 선으로 된, "
    # endregion

    # region Class: trunk shade
    print("tree_model_trunk_shade")
    pred = models['trunk_shade'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2] and pred[0, 0] > pred[0, 3]:
        trunk_str += "전체 명암이 있는, "
    elif pred[0, 1] > pred[0, 2] and pred[0, 1] > pred[0, 0] and pred[0, 1] > pred[0, 3]:
        trunk_str += "우측의 그림자, "
    elif pred[0, 2] > pred[0, 0] and pred[0, 2] > pred[0, 1] and pred[0, 2] > pred[0, 3]:
        trunk_str += "좌측의 그림자, "
    # endregion

    # region Class: trunk tilt
    print("tree_model_trunk_tilt")
    pred = models['trunk_tilt'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2]:
        trunk_str += "우측으로 기울어진, "
    elif pred[0, 1] > pred[0, 2]:
        trunk_str += "좌측으로 기울어진, "
    # endregion

    # region Class: trunk pattern
    print("tree_model_trunk_pattern")
    pred = models['trunk_pattern'].predict(bottle_resized2)

    if pred[0, 1] > pred[0, 0] and pred[0, 1] > pred[0, 2]:
        trunk_str += "둥근 껍질 무늬, "
    elif pred[0, 2] > pred[0, 1]:
        trunk_str += "긁힌 무늬, "
    # endregion

    # region Class: low branch
    print("tree_model_low_branch")
    pred = models['low_branch'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1]:
        trunk_str += "가지가 있는, "
    # endregion

    if trunk_str != "나무기둥: ":
        final_sentence += trunk_str

    return path_details, path_detected, final_sentence


def test_cat(models, img_dir):
    """
        Test cat image with models.
        1. Detect regions (cat, head, body)
        2. Crop and classify with classification model
        3. Get image attributes
        :param img_dir: image from web page
        :return: cat image, trunk image, head image, body image with detected region, result sentence
        """
    dest = img_dir.replace(os.path.basename(img_dir), '')
    path_details = {}

    models['default'].cfg.DATASETS.TEST = (img_dir,)
    test_metadata = MetadataCatalog.get(img_dir)

    print("test cat!!!!!!!!")
    print("cat file: " + img_dir)
    filename = img_dir.split('/')[-1]

    im = cv2.imread(img_dir)

    # region Detection Part
    outputs = models['default'](im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata,
                   scale=0.5
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Fix cat image size: (560, 400)
    resized_img = cv2.resize(out.get_image()[:, :, ::-1], dsize=(560, 400), interpolation=cv2.INTER_AREA)
    name = "{}_{}.jpg".format(filename.split('.')[0], "detected")
    path_detected = os.path.join(dest, name)
    cv2.imwrite(path_detected, resized_img)

    boxes = {}
    array = []
    # label name
    label_array = []

    for label in outputs["instances"].to("cpu").pred_classes:
        if label.item() == 0:
            label = "body"
        elif label.item() == 1:
            label = "cat"
        elif label.item() == 2:
            label = "head"

        label_array.append(label)

    for coordinates in outputs["instances"].to("cpu").pred_boxes:
        coordinates_array = []
        for k in coordinates:
            coordinates_array.append(int(k))
        array.append(coordinates_array)

    for i in range(len(label_array)):
        boxes[label_array[i]] = array[i]

    print("STEP1 BOXES: ", boxes)

    # crop image
    for k, v in boxes.items():
        crop_img = im[v[1]:v[3], v[0]:v[2], :]
        # plt.imshow(crop_img, interpolation='nearest')
        # plt.show()

        ratio = 350.0 / crop_img.shape[1]
        dim = (350, int(crop_img.shape[0] * ratio))

        resized_img = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
        name = "{}_{}.jpg".format(filename.split('.')[0], k)
        path_details[k] = os.path.join(dest, name)
        cv2.imwrite(path_details[k], resized_img)

    # endregion

    # From this line, we get image attributes with classification models
    # Check attributes and see which brain is more active. Left brain or right brain?

    # First, check cat attributes
    left = 0
    right = 0

    bottle_resized = resize(im, (224, 224))
    bottle_resized = np.expand_dims(bottle_resized, axis=0)

    # region Class: conceptual
    pred = models['concept'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        print("RIGHT HEMISPHERE")
        right += 1
    else:
        print("LEFT HEMISPHERE")
        left += 1
    # endregion

    # region Class: movement
    pred = models['movement'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        print("LEFT HEMISPHERE")
        left += 1
    else:
        print("RIGHT HEMISPHERE")
        right += 1

    # Second, check image colorfulness
    c = image_colorfulness(im)
    if c > 50:
        print("COLORFUL CAT - RIGHT HEMISPHERE")
        right += 1
    else:
        print("COLORLESS CAT - LEFT HEMISPHERE")
        left += 1

    print("RIGHT: ", right)
    print("LEFT: ", left)

    # Draw pie graph with the result
    plt.rcParams['figure.figsize'] = [12, 8]

    group_names = ['right brain', 'left brain']
    group_sizes = [right, left]
    group_colors = ['red', 'blue']

    plt.pie(group_sizes,
            labels=group_names,
            colors=group_colors,
            autopct='%1.2f%%',  # second decimal place
            shadow=True,
            startangle=90,
            textprops={'fontsize': 14})  # text font size

    name = "{}_{}.jpg".format(filename.split('.')[0], "plot")
    path_plot = os.path.join(dest, name)

    plt.axis('equal')  # equal length of X and Y axis
    plt.title('Your brain is?', fontsize=20)
    plt.savefig(path_plot)

    # plt.show()
    plt.close()

    # Done! Return images, pie graph and result sentence
    final_sentence = ""
    if right > left:
        final_sentence = "당신은 우뇌입니다"
    elif left > right:
        final_sentence = "당신은 좌뇌입니다"

    # endregion

    # return path_list['cat'], path_list['head'], path_list['body'], path_detected, result, path_plot
    return path_details, path_detected, path_plot, final_sentence