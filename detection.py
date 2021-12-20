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
    cfg.merge_from_file(
        # "/home/jang/anaconda3/envs/drawing_env/lib/python3.7/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        env_path.replace("/bin/python",
                         "/lib/python3.7/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )
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


def test_tree(models, img_dir):
    """
    Test tree image with models.
    1. Detect regions (branches, trunk, root)
    2. Crop and classify with classification model
    3. Get image attributes
    :param img_dir: image from web page
    :return: branch image, trunk image, root image, tree image with detected region , result sentence
    """
    dest = img_dir.replace(os.path.basename(img_dir), '')
    path_list = {}

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
    path_list['detected'] = os.path.join(dest, name)
    cv2.imwrite(path_list['detected'], resized_img)

    # Analyzing if there's a lot of space in the image
    # 1. Convert image to GRAYSCALE
    # 2. Count pixel value
    img1 = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)

    scale_percent = 50
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', img.shape)

    # total = img.shape[0] * img.shape[1]
    # print(total)

    # x1 = img.shape[0] / 5
    # x2 = img.shape[0] - x1
    y1 = img.shape[1] / 5
    y2 = img.shape[1] - y1

    # total = x1 * img.shape[1]
    # count = getWhitePercent(img, total, 0, x1, 0, img.shape[1])
    # count = getWhitePercent(img, total, x2, img.shape[0], 0, img.shape[1])

    total = y1 * img.shape[0]
    # count = getWhitePercent(img, total, 0, img.shape[0], 0, y1)
    count = getWhitePercent(img, total, 0, img.shape[0], y2, img.shape[1])

    if count >= 3:
        print('image is dense')
        attr1 = "공간 밀도"
        attr2 = "꽉찬"
        adj1 = "근면하고"
        adj2 = "본능에 끌리지 않고"
        adj3 = "갇힌 감정 상태"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence00 = adj_sentence(attr1, attr2, adj, adj3)

    else:
        print('image is empty')
        attr1 = "공간 밀도"
        attr2 = "비어있는"
        adj1 = "민감하고"
        adj2 = "겸손하고"
        adj3 = "정신적으로 어떻게 행동해야 할지 모르는 상태"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence00 = adj_sentence(attr1, attr2, adj1, adj3)

    # The last sentence to be returned
    # Can see this 'final_sentence' through web
    final_sentence = sentence00 + "\r\n"

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

    # {'branches': [623, 826, 2086, 1853], 'trunk': [1075, 1796, 1459, 2269], 'roots': [733, 2228, 1549, 2439]}
    print("STEP1 BOXES: ", boxes)

    # Crop Image
    img_array = []
    for k, v in boxes.items():
        crop_img = im[v[1]:v[3], v[0]:v[2], :]
        # plt.imshow(crop_img, interpolation='nearest')
        # plt.show()

        ratio = 350.0 / crop_img.shape[1]
        dim = (350, int(crop_img.shape[0] * ratio))

        detected_file = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
        name = "{}_{}.jpg".format(filename.split('.')[0], k)
        path_list[k] = os.path.join(dest, name)
        cv2.imwrite(path_list[k], detected_file)
        img_array.append(crop_img)

    # From this line, we get image attributes with classification models

    # First, check if there is 'root'
    # print("IMG_ARRAY LEN", len(img_array))  # 3(if root exists), 2(if root does not exist)
    if len(img_array) == 3:
        attr1 = "뿌리"
        attr2 = "뿌리가 보이는 상태인"
        adj1 = "원시성이 있고"
        adj2 = "전통과의 결부가 보여지고"
        adj3 = "정확성"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence0 = adj_sentence(attr1, attr2, adj, adj3)

        final_sentence = final_sentence + sentence0 + "\r\n"

    # Second, check tree attributes for branches(crown)
    img = img_array[0]

    bottle_resized = resize(img, (224, 224))
    bottle_resized = np.expand_dims(bottle_resized, axis=0)
    # endregion

    # region Class: crown shape
    print("tree_model_crown_shape")
    pred = models['crown_shape'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2]:
        attr1 = "관"
        attr2 = "아케이드 모양인"

        adj1 = "감수성이 있고"
        adj2 = "예의 바르고"
        adj3 = "의무감"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence1 = adj_sentence(attr1, attr2, adj, adj3)
        final_sentence = final_sentence + sentence1 + "\r\n"

    elif pred[0, 1] > pred[0, 2]:
        attr1 = "관"
        attr2 = "공 모양인"

        adj1 = "에너지가 부족하고"
        adj2 = "구성 감각이 결여되어있고"
        adj3 = "텅빈 마음"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence1 = adj_sentence(attr1, attr2, adj, adj3)
        final_sentence = final_sentence + sentence1 + "\r\n"
    # endregion

    # region Class: crown shade
    print("tree_model_crown_shade")
    pred = models['crown_shade'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        attr1 = "관"
        attr2 = "그림자 진"

        adj1 = "분위기에 좌우되고"
        adj2 = "정확성의 결여되어있고"
        adj3 = "부드러움"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence2 = adj_sentence(attr1, attr2, adj, adj3)

        final_sentence = final_sentence + sentence2 + "\r\n"
    # endregion

    # region Class: crown fruit
    print("tree_model_crown_fruit")
    pred = models['crown_fruit'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        attr1 = "관"
        attr2 = "과일이 매달려 있는"
        adj1 = "발달이 지체되어있고"
        adj2 = "자기 표현 능력이 결여되어있고"
        adj3 = "독립심의 결여"

        sentence3 = adj_sentence(attr1, attr2, adj, adj3)

        final_sentence = final_sentence + sentence3 + "\r\n"
    # endregion

    # region Class: cut branch
    print("tree_model_cut_branch")
    pred = models['cut_branch'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        attr1 = "가지"
        attr2 = "잘려있는"

        adj1 = "살려는 의지가 있고"
        adj2 = "억제되어있고"
        adj3 = "저항력"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence4 = adj_sentence(attr1, attr2, adj, adj3)

        final_sentence = final_sentence + sentence4 + "\r\n"

    # Last, check tree attributes for trunk
    img2 = img_array[1]
    bottle_resized2 = resize(img2, (224, 224))
    bottle_resized2 = np.expand_dims(bottle_resized2, axis=0)
    # endregion

    # region Class: trunk shape
    print("tree_model_trunk_shape")
    pred = models['trunk_shape'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1]:
        # attr = "Trunk base: "
        attr1 = "나무기둥"
        attr2 = "양쪽으로 넓은 모양인"

        adj1 = "봉쇄적 사고가 있고"
        adj2 = "이해가 느리고"
        adj3 = "학습곤란"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence5 = adj_sentence(attr1, attr2, adj, adj3)

    else:
        attr1 = "나무기둥"
        attr2 = "직선적인 모양인"

        adj1 = "규범적이고"
        adj2 = "고집이 세고"
        adj3 = "냉정함"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence5 = adj_sentence(attr1, attr2, adj, adj3)

    if sentence5:
        final_sentence = final_sentence + sentence5 + "\r\n"
    # endregion

    # region Class: trunk wave
    print("tree_model_trunk_wave")
    pred = models['trunk_wave'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1]:
        attr1 = "나무기둥"
        attr2 = "구불거리는 모양인"

        adj1 = "생동감이 있는"
        adj2 = "적응력이 큰"
        adj3 = "생기"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence6 = adj_sentence(attr1, attr2, adj, adj3)

        final_sentence = final_sentence + sentence6 + "\r\n"
    # endregion

    # region Class: trunk lines
    print("tree_model_trunk_lines")
    pred = models['trunk_lines'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1]:
        attr1 = "나무기둥"
        attr2 = "흩어진 선으로 이루어진"

        adj1 = "예민하고"
        adj2 = "감정이입을 강하게 하는 경향이 있고"
        adj3 = "민감성"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence7 = adj_sentence(attr1, attr2, adj, adj3)

        final_sentence = final_sentence + sentence7 + "\r\n"
    # endregion

    # region Class: trunk shade
    print("tree_model_trunk_shade")
    pred = models['trunk_shade'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2] and pred[0, 0] > pred[0, 3]:
        # attr = "Trunk full shade: "
        attr1 = "나무기둥"
        attr2 = "전체에 명암이 있는"

        adj1 = "수동적이고"
        adj2 = "강박적이고"
        adj3 = "불안정감"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence8 = adj_sentence(attr1, attr2, adj, adj3)
        final_sentence = final_sentence + sentence8 + "\r\n"

    elif pred[0, 1] > pred[0, 2] and pred[0, 1] > pred[0, 0] and pred[0, 1] > pred[0, 3]:
        # attr = "Trunk right shade: "
        attr1 = "나무기둥"
        attr2 = "오른쪽에 그림자가 있는"

        adj1 = "접촉할 능력이 있고"
        adj2 = "접촉할 능력이 있고"
        adj3 = "적응력"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence8 = adj_sentence(attr1, attr2, adj, adj3)
        final_sentence = final_sentence + sentence8 + "\r\n"

    elif pred[0, 2] > pred[0, 0] and pred[0, 2] > pred[0, 1] and pred[0, 2] > pred[0, 3]:
        # attr = "Trunk left shade: "
        attr1 = "나무기둥"
        attr2 = "왼쪽에 그림자가 있는"

        adj1 = "외향적이고"
        adj2 = "억제하는 경향이 있고"
        adj3 = "민감성"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence8 = adj_sentence(attr1, attr2, adj, adj3)
        final_sentence = final_sentence + sentence8 + "\r\n"
    # endregion

    # region Class: trunk tilt
    print("tree_model_trunk_tilt")
    pred = models['trunk_tilt'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2]:
        # attr = "Trunk right tilt: "
        attr1 = "나무기둥"
        attr2 = "오른쪽으로 기울어진"

        adj1 = "집중을 잘하고"
        adj2 = "유혹에 빠지기 쉽고"
        adj3 = "민감성"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence9 = adj_sentence(attr1, attr2, adj, adj3)
        final_sentence = final_sentence + sentence9 + "\r\n"

    elif pred[0, 1] > pred[0, 2]:
        # attr = "Trunk left tilt: "
        attr1 = "나무기둥"
        attr2 = "왼쪽으로 기울어진"

        adj1 = "도전적이고"
        adj2 = "감정을 억누르는 경향이 있고"
        adj3 = "방어적 태도"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence9 = adj_sentence(attr1, attr2, adj, adj3)
        final_sentence = final_sentence + sentence9 + "\n"
    # endregion

    # region Class: trunk pattern
    print("tree_model_trunk_pattern")
    pred = models['trunk_pattern'].predict(bottle_resized2)

    if pred[0, 1] > pred[0, 0] and pred[0, 1] > pred[0, 2]:
        # attr = "Trunk round pattern: "
        attr1 = "나무기둥"
        attr2 = "둥근 나무 껍질 무늬가 있는"

        adj1 = "접촉을 위한 준비 능력이 있고"
        adj2 = "접촉을 위한 준비 능력이 있고"
        adj3 = "자발적 적응 능력"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence10 = adj_sentence(attr1, attr2, adj, adj3)
        final_sentence = final_sentence + sentence10 + "\r\n"

    elif pred[0, 2] > pred[0, 1]:
        # attr = "Trunk scratch pattern: "
        attr1 = "나무기둥"
        attr2 = "긁힌 모양의 무늬가 있는"

        adj1 = "냉정하고"
        adj2 = "규범적이고"
        adj3 = "센 고집"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence10 = adj_sentence(attr1, attr2, adj, adj3)
        final_sentence = final_sentence + sentence10 + "\n"
    # endregion

    # region Class: low branch
    print("tree_model_low_branch")
    pred = models['low_branch'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1]:
        # attr = "Low branch: "
        attr1 = "나무기둥"
        attr2 = "가지가 있는"

        adj1 = "신뢰성이 없고"
        adj2 = "행동이 어린 아이 같고"
        adj3 = "부분적 발달 억제"

        adj_list = [adj1, adj2]
        adj = random.choice(adj_list)

        sentence11 = adj_sentence(attr1, attr2, adj, adj3)

        final_sentence = final_sentence + sentence11 + "\r\n"
    # endregion

    return path_list['branches'], path_list['trunk'], path_list['roots'], path_list['detected'], final_sentence


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
    path_list = {}

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
    path_list['detected'] = os.path.join(dest, name)
    cv2.imwrite(path_list['detected'], resized_img)

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
        path_list[k] = os.path.join(dest, name)
        cv2.imwrite(path_list[k], resized_img)

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
    path_list['plot'] = os.path.join(dest, name)

    plt.axis('equal')  # equal length of X and Y axis
    plt.title('Your brain is?', fontsize=20)
    plt.savefig(path_list['plot'])

    # plt.show()
    plt.close()

    # Done! Return images, pie graph and result sentence
    result = ""
    if right > left:
        result = "당신은 우뇌입니다"
    elif left > right:
        result = "당신은 좌뇌입니다"

    # endregion

    return path_list['cat'], path_list['head'], path_list['body'], path_list['detected'], result, path_list['plot']