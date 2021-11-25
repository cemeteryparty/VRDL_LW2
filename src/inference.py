from tensorflow import keras
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import resize_image
from scipy.io import loadmat
from scipy.io import savemat
import pandas as pd
import numpy as np
import argparse
import json
import cv2
import sys
import os

BASE_DIR = "/mnt/left/phlai_DATA/vrdl5008/"


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Inference dataset by specific model"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--model-path', help='Inference model path'
    )
    parser.add_argument(
        '--image-min-side',
        help='Rescale the image so the smallest side is min_side.',
        type=int, default=800
    )
    parser.add_argument(
        '--image-max-side',
        help='Rescale the image if the largest side is larger than max_side.',
        type=int, default=1333
    )

    return parser.parse_args(args)


def inferene_image(model, image, min_side=800, max_side=1333):
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side=min_side, max_side=max_side)
    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(image, axis=0)
    )
    boxes /= scale

    return boxes.squeeze(), scores.squeeze(), labels.squeeze()


def re_inference(model, image, min_side=800, max_side=1333, k=2):
    im_width, im_height = image.shape[1] // k, image.shape[0] // k
    boxes_list, score_list, label_list = None, None, None
    for i in range(k):
        for j in range(k):
            part = image[
                i * im_height:(i + 1) * im_height,
                j * im_width:(j + 1) * im_width
            ]
            boxes, scores, labels = inferene_image(
                model, part, min_side=min_side, max_side=max_side
            )
            boxes[:, 0] = boxes[:, 0] + j * im_width
            boxes[:, 2] = boxes[:, 2] + j * im_width
            boxes[:, 1] = boxes[:, 1] + i * im_height
            boxes[:, 3] = boxes[:, 3] + i * im_height
            if boxes_list is None:
                boxes_list, score_list, label_list = boxes, scores, labels
            else:
                boxes_list = np.append(boxes_list, boxes, axis=0)
                score_list = np.append(score_list, scores, axis=0)
                label_list = np.append(label_list, labels, axis=0)
    order = score_list.argsort()[::-1]
    return boxes_list[order], score_list[order], label_list[order]


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print(args)
    # raise Exception("Debug Checkpoint")

    model = models.load_model(args.model_path)
    fnames = os.listdir("test")
    result_to_json = []

    for fname in fnames:
        image_id = int(fname[:-4])

        image = cv2.imread("test/" + fname)

        boxes, scores, labels = inferene_image(
            model, image.copy(),
            min_side=args.image_min_side, max_side=args.image_max_side
        )
        """ if no confidence, reinference by partition of image """
        k = 2
        while scores[0] < 0.6:
            if k > 5:
                break
            boxes_, scores_, labels_ = re_inference(
                model, image.copy(),
                min_side=args.image_min_side, max_side=args.image_max_side, k=k
            )
            boxes = np.append(boxes, boxes_, axis=0)
            scores = np.append(scores, scores_, axis=0)
            labels = np.append(labels, labels_, axis=0)
            order = scores.argsort()[::-1]
            boxes, scores, labels = boxes[order], scores[order], labels[order]
            k += 1

        """ add each detection box infomation into list """
        for i in range(labels.size):
            if i > 50 or labels[i] < 0:
                break
            x = boxes[i][0]
            y = boxes[i][1]
            w = boxes[i][2] - boxes[i][0]
            h = boxes[i][3] - boxes[i][1]
            """ save info in dict
                json can't serialize np.float32 and np.int32
                use float and int to transform to python object
            """
            det_box_info = {}
            # int: image id
            det_box_info["image_id"] = image_id
            # list: [left_x, top_y, width, height]
            det_box_info["bbox"] = [float(x), float(y), float(w), float(h)]
            # float [0.0, 1.0]: confidence of the bbox
            det_box_info["score"] = float(scores[i])
            # int: label
            det_box_info["category_id"] = int(labels[i])

            result_to_json.append(det_box_info)

    json_object = json.dumps(result_to_json, indent=4)
    with open("output/answer.json", "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    main()
