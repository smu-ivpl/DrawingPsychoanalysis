import os
import argparse

import warnings

# Import MaskRCNN
from mrcnn import model as modellib
from text_detection import TextConfig

# Import demo
from demo import run_mrcnn

# Computer Setting
warnings.simplefilter(action='ignore', category=Warning)


def set_detection(opt):
    class InferenceConfig(TextConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=opt)

    # Select weights file to load
    weights_path = opt.weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    return config, model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="OCR!"
                    "Process: Detection --> Recognition"
                    "Dataset: Life Space or Social Atom")

    parser.add_argument("command",
                        default="<command>",
                        help="choose option: demo")
    parser.add_argument("--drawing", required=False,
                        metavar="drawing",
                        default="life_space",
                        help='drawing data: life_space|social_atom')

    """ Text Detection Configuration"""
    parser.add_argument('--weights', required=False,
                        metavar='coco',
                        default='life_space_detection_model/mask_rcnn_text3_0100.h5',
                        help="Path to weights .h5 file or 'coco'")

    args = parser.parse_args()

    # Check drawing type
    print("Drawing: ", args.drawing)

    # Demo
    if args.command == "demo":
        print("Start: ", args.command)

        # Detection Setting
        config, detection_model = set_detection(args)

        # Start
        image_path = args.drawing + '_demo/'
        dirs = sorted(os.listdir(image_path))
        print(dirs)
        images = [file for file in dirs if file.endswith('.jpg')]

        for img in images:
            img_name = os.path.join(image_path, img)
            only_name, _ = os.path.splitext(img)
            img_jpg = only_name + '.jpg'

            run_mrcnn(detection_model=detection_model,
                      padding=5, image_path=image_path, image_name=img_name, img_file_name=img_jpg)
