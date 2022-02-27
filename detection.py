from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from tensorflow import keras
# import keras
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
# Import MaskRCNN
from mrcnn import model as modellib
from mrcnn import visualize
from text_detection import TextConfig
from demo import run_mrcnn

import skimage.draw

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

from kor_string import all_kor_string, family_string

from dotted.collection import DottedDict
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matplotlib.use('agg')
count = 0

opt = DottedDict({
        "image_folder": "",
        "workers": 4,
        "batch_size": 192,
        "saved_model": "ckpt/model_life_space/best_accuracy.pth",
        "batch_max_length": 25,
        "imgH": 64,
        "imgW": 200,
        "rgb": False,
        "character": all_kor_string,
        "sensitive": False,
        "PAD": False,
        "Transformation": "TPS",
        "FeatureExtraction": "ResNet",
        "SequenceModeling": "BiLSTM",
        "Prediction": "CTC",
        "num_fiducial": 20,
        "input_channel": 1,
        "output_channel": 512,
        "hidden_size": 256,
        "num_class": -1,
        "num_gpu": torch.cuda.device_count(),
        "converter": None
    })

def init():
    cfg = get_cfg()

    import shutil
    env_path = shutil.which('python')
    cfg.merge_from_file(env_path.replace("/bin/python",
        "/lib/python3.7/site-packages/detectron2/model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "ckpt/model_final_f6e8b1.pkl"
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

    cfg.MODEL.WEIGHTS = os.path.join("ckpt/model_detection_tree/model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model

    models = {}
    models['tree'] = {}
    models['cat'] = {}
    models['life'] = {}

    models['tree']['default'] = DefaultPredictor(cfg)

    models['tree']['crown_shape'] = keras.models.load_model(
        "ckpt/model_classification_tree/crown_shape_model.h5")
    models['tree']['crown_shade'] = keras.models.load_model(
        "ckpt/model_classification_tree/crown_shade_model.h5")
    models['tree']['crown_fruit'] = keras.models.load_model(
        "ckpt/model_classification_tree/fruit_model.h5")
    models['tree']['cut_branch'] = keras.models.load_model(
        "ckpt/model_classification_tree/cut_branch_model.h5")
    models['tree']['trunk_shape'] = keras.models.load_model(
        "ckpt/model_classification_tree/trunk_shape_model.h5")
    models['tree']['trunk_wave'] = keras.models.load_model(
        "ckpt/model_classification_tree/trunk_wave_model.h5")
    models['tree']['trunk_lines'] = keras.models.load_model(
        "ckpt/model_classification_tree/trunk_lines_model.h5")
    models['tree']['trunk_shade'] = keras.models.load_model(
        "ckpt/model_classification_tree/trunk_shade_model.h5")
    models['tree']['trunk_tilt'] = keras.models.load_model(
        "ckpt/model_classification_tree/trunk_tilt_model.h5")
    models['tree']['trunk_pattern'] = keras.models.load_model(
        "ckpt/model_classification_tree/trunk_pattern_model.h5")
    models['tree']['low_branch'] = keras.models.load_model(
        "ckpt/model_classification_tree/low_branch_model.h5")

    cfg.MODEL.WEIGHTS = os.path.join("ckpt/model_detection_cat/model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

    models['cat']['default'] = DefaultPredictor(cfg)
    models['cat']['concept'] = keras.models.load_model("ckpt/model_classification_cat/concept_model.h5")
    models['cat']['movement'] = keras.models.load_model(
        "ckpt/model_classification_cat/movement_model.h5")

    # Lifespace Detection Setting

    # Parse command line arguments

    weights = 'ckpt/model_life_space/mask_rcnn_text_0001.h5'
    _, models['life']['mrcnn'] = set_detection(weights)

    cudnn.benchmark = True
    cudnn.deterministic = True

    opt.converter = CTCLabelConverter(all_kor_string)
    opt.num_class = len(opt.converter.character)

    print('deep-text-recognition-benchmark parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    models['life']['dtrb'] = Model(opt).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    state_dict = torch.load(opt.saved_model, map_location=device)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        new_state_dict[new_k] = v

    models['life']['dtrb'].load_state_dict(new_state_dict)

    return models


def set_detection(weights):
    class InferenceConfig(TextConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    #config.display()

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=weights)

    # Select weights file to load
    weights_path = weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    return config, model


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
    path_details = OrderedDict()

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
        # 꽉 찬 공간밀도
        results = "근면한, 본능에 끌리지 않는, 갇힌 감정 상태, "
    else:
        # 비어있는 공간밀도
        results = "민감한, 겸손한, 정신적으로 어찌할 바를 모르는 상태, "

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
        # 뿌리 보임
        sentence0 = "\nroots: 원시성이 있는, 전통과의 결부가 보이는, 정확성 있는, "
        final_sentence = final_sentence + sentence0

    img = img_array[0]

    bottle_resized = resize(img, (224, 224))
    bottle_resized = np.expand_dims(bottle_resized, axis=0)
    # endregion

    # region Class: crown shape
    print("tree_model_crown_shape")
    pred = models['crown_shape'].predict(bottle_resized)

    crown_str = "\nbranches: "
    if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2]:
        # 아케이드 모양
        crown_str += "감수성이 있는, 예의 바른, 의무감, "

    elif pred[0, 1] > pred[0, 2]:
        # 공 모양
        crown_str += "에너지가 부족한, 구성 감각이 결여된, 텅빈 마음, "
    # endregion

    # region Class: crown shade
    print("tree_model_crown_shade")
    pred = models['crown_shade'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        # 그림자 있는
        crown_str += "분위기에 좌우되는, 정확성이 결여된, 부드러움, "
    # endregion

    # region Class: crown fruit
    print("tree_model_crown_fruit")
    pred = models['crown_fruit'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        # 과일 달린
        crown_str += "발달이 지체된, 자기 표현 능력이 결여된, 독립심이 결여된, "
    # endregion

    if crown_str != "\nbranches: ":
        final_sentence += crown_str

    # region Class: cut branch
    print("tree_model_cut_branch")
    pred = models['cut_branch'].predict(bottle_resized)

    if pred[0, 0] > pred[0, 1]:
        # 잘린 나뭇가지
        final_sentence += "생존 의지가 있는, 억제되어 있는, 저항력, "

    # Last, check tree attributes for trunk
    img2 = img_array[1]
    bottle_resized2 = resize(img2, (224, 224))
    bottle_resized2 = np.expand_dims(bottle_resized2, axis=0)
    # endregion

    # region Class: trunk shape
    print("tree_model_trunk_shape")
    pred = models['trunk_shape'].predict(bottle_resized2)

    trunk_str = "\ntrunk: "
    if pred[0, 0] > pred[0, 1]:
        # 양쪽으로 넓은
        trunk_str += "봉쇄적 사고가 있는, 이해가 느린, 학습곤란, "
    else:
        # 직선 모양의
        trunk_str += "규범적인, 고집이 센, 냉정함, "
    # endregion

    # region Class: trunk wave
    print("tree_model_trunk_wave")
    pred = models['trunk_wave'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1]:
        # 구불거리는 형태
        trunk_str += "생동감이 있는, 적응력이 큰, 생기있는, "
    # endregion

    # region Class: trunk lines
    print("tree_model_trunk_lines")
    pred = models['trunk_lines'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1]:
        # 흩어진 선으로 된
        trunk_str += "예민한, 감정이입이 강한 경향, 민감성, "
    # endregion

    # region Class: trunk shade
    print("tree_model_trunk_shade")
    pred = models['trunk_shade'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2] and pred[0, 0] > pred[0, 3]:
        # 전체 명암이 있는
        trunk_str += "수동적인, 강박적인, 불안정감, "
    elif pred[0, 1] > pred[0, 2] and pred[0, 1] > pred[0, 0] and pred[0, 1] > pred[0, 3]:
        # 우측의 그림자
        trunk_str += "접촉할 능력이 있는, 적응력, "
    elif pred[0, 2] > pred[0, 0] and pred[0, 2] > pred[0, 1] and pred[0, 2] > pred[0, 3]:
        # 좌측의 그림자
        trunk_str += "외향적인, 억제하는 경향, 민감성, "
    # endregion

    # region Class: trunk tilt
    print("tree_model_trunk_tilt")
    pred = models['trunk_tilt'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1] and pred[0, 0] > pred[0, 2]:
        # 우측으로 기울어진
        trunk_str += "집중을 잘하는, 유혹에 빠지기 쉬운, "
    elif pred[0, 1] > pred[0, 2]:
        # 좌측으로 기울어진
        trunk_str += "도전적인, 감정을 억누르는 경향이 있는, 방어적 태도, "
    # endregion

    # region Class: trunk pattern
    print("tree_model_trunk_pattern")
    pred = models['trunk_pattern'].predict(bottle_resized2)

    if pred[0, 1] > pred[0, 0] and pred[0, 1] > pred[0, 2]:
        # 둥근 껍질 무늬
        trunk_str += "접촉을 위한 준비 능력이 있는, 자발적 적응 능력, "
    elif pred[0, 2] > pred[0, 1]:
        # 긁힌 무늬
        trunk_str += "냉정한, 규범적인, 고집 센, "
    # endregion

    # region Class: low branch
    print("tree_model_low_branch")
    pred = models['low_branch'].predict(bottle_resized2)

    if pred[0, 0] > pred[0, 1]:
        # 가지가 있는
        trunk_str += "신뢰성이 없는, 행동이 어린 아이 같은, 부분적 발달 억제, "
    # endregion

    if trunk_str != "\ntrunk: ":
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
    path_details = OrderedDict()

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

        ratio = 350.0 / crop_img.shape[1]
        dim = (350, int(crop_img.shape[0] * ratio))

        resized_img = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)
        name = "{}_{}.jpg".format(filename.split('.')[0], k)
        path_details[k] = os.path.join(dest, name)
        cv2.imwrite(path_details[k], resized_img)

    # endregion

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

    percent = left / 10. * 100.
    group_names = ['left brain: {}%'.format(percent), ' ']
    group_sizes = [left, 10 - left]
    group_colors = ['red', 'white']

    plt.subplot(1, 2, 1)
    plt.pie(group_sizes,
            labels=group_names,
            colors=group_colors,
            shadow=True,
            startangle=90,
            textprops={'fontsize': 14})  # text font size
    plt.axis('equal')  # equal length of X and Y axis

    percent = right / 10. * 100.
    group_names = [' ', 'right brain: {}%'.format(percent)]
    group_sizes = [10 - right, right]
    group_colors = ['white', 'blue']

    plt.subplot(1, 2, 2)
    plt.pie(group_sizes,
            labels=group_names,
            colors=group_colors,
            shadow=True,
            startangle=90,
            textprops={'fontsize': 14})  # text font size
    plt.axis('equal')  # equal length of X and Y axis

    name = "{}_{}.jpg".format(filename.split('.')[0], "plot")
    path_plot = os.path.join(dest, name)

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


def test_life(models, img_path):

    dir_path = os.path.dirname(img_path)
    opt.image_folder = os.path.join(dir_path, 'cropped')
    os.mkdir(opt.image_folder)

    bbox, result = run_mrcnn(detection_model=models['mrcnn'], padding=5, image_path=img_path)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    crop_names = []
    final_results = OrderedDict()
    final_results["Detection"] = os.path.join(dir_path, "bb_visualize.jpg")

    # predict
    models['dtrb'].eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            cropped_image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            preds = models['dtrb'](cropped_image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = opt.converter.decode(preds_index, preds_size)

            log = open(os.path.join(os.path.dirname(img_path), f'log_demo_result.txt'), 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):

                final_results[pred] = img_name

                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                crop_names.append(pred)
                print(f'{os.path.basename(img_name):25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{os.path.basename(img_name):25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()

    image = skimage.io.imread(img_path)
    image = skimage.color.gray2rgb(image)
    visualize.display_instances_with_grid(final_results["Detection"],
                                          image, bbox, result['masks'], result['class_ids'],
                                          crop_names, result['scores'], title=img_path)

    return final_results
