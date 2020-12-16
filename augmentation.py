import json
from copy import deepcopy

import albumentations as A
import cv2

from label import Label
import matplotlib.pyplot as plt
import numpy as np


def generate_sample(image, labels):
    transform = A.Compose([
        A.RandomResizedCrop(width=image.shape[1],
                            height=image.shape[0],
                            scale=(0.4, 0.9),
                            ratio=(image.shape[1] / image.shape[0], image.shape[1] / image.shape[0]),
                            interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        #  A.RandomBrightnessContrast(p=0.4),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=10, min_visibility=0.6))

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = transform(image=image, bboxes=labels)
    return transformed["image"], transformed["bboxes"]


def generate_augmentations_for_image(image, labels, max_count, max_iters=1000):
    res_im = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB), ]
    res_la = [labels, ]
    label_cpy = []
    for i in range(len(labels)):
        label_cpy.append((*labels[i], "some fake label"))
    for i in range(max_iters):
        new_image, new_labels = generate_sample(deepcopy(image), label_cpy)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        for i in range(len(new_labels)):
            new_labels[i] = list(int(float(new_labels[i][j])) for j in range(4))

        if len(new_labels) > 0 and len(new_labels) >= len(labels) * 0.4:
            res_im.append(new_image)
            res_la.append(np.asarray(new_labels))
        if len(res_im) >= max_count:
            break
    return res_im, res_la


if __name__ == '__main__':
    image = cv2.imread("data\CW\original\clone-captain-rex-main_7f7a2ec0.jpeg")
    with open("data\CW\original\clone-captain-rex-main_7f7a2ec0.json") as f:
        data = json.load(f)
        shapes = [
            Label((*shape["points"][0], *shape["points"][1])).pascal_voc_with_name() for shape in data["shapes"]
        ]
        r = [(image, shapes)]
        for i in range(5):
            r.append(generate_sample(image, shapes))

        for i in range(len(r)):
            p = plt.subplot(1, len(r), 1 + i)
            im, labels = r[i]
            for label in labels:
                label = [int(float(x)) for x in label[:4]]
                im[label[1]:label[3], label[0]:label[2], :] = np.asarray((255, 0, 0))

            p.imshow(im)
        plt.show()
