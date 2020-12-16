import json
from copy import deepcopy
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

from label import Label


class ArrayLoader:
    def __init__(self, files_list):
        self.files_list = files_list

    def get(self, indices, shape, with_names=False):
        res_p = []
        names = []
        for i in tqdm(indices):
            res_p.append(load_image(Path(self.files_list[i]), shape, load_json=False))
            names.append(self.files_list[i])

        res = np.asarray(res_p)
        if (with_names):
            return res, names
        else:
            return res

    def get_index(self, i: int, shape):
        return self.get((i,), shape)[0]

    def get_all(self, shape):
        return self.get(range(0, len(self.files_list)), shape)

    def size(self):
        return len(self.files_list)


def load_image(path, shape=None, load_json=True):
    arr = np.array(Image.open(str(path)))
    target_size = deepcopy(arr.shape)
    if arr.shape[1] / arr.shape[0] == 1280 / 720:
        res_dx = int(arr.shape[0] * ((720 - 544) / 2) / 720)
        arr = arr[res_dx: arr.shape[0] - res_dx, :, :]

    if arr.shape != (544, 1280, 3):
        arr = cv2.resize(arr, dsize=(1280, 544), interpolation=cv2.INTER_AREA)

    if shape is not None and arr.shape != shape:
        arr = cv2.resize(arr, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_AREA)

    if load_json:
        return arr, load_json_for_image(str(path.with_suffix(".json")), shape, target_size)
    else:
        return arr


def load_json_for_image(path, shape, init_shape):
    with open(path) as f:
        data = json.load(f)
        labels = []
        for shap in data["shapes"]:
            labels.append(Label((*shap["points"][0], *shap["points"][1])).pascal_voc())
        labels = np.asarray(labels)

        if init_shape[1] / init_shape[0] == 1280 / 720:
            res_dx = int(init_shape[0] * ((720 - 544) / 2) / 720)
            init_shape = (init_shape[0] - 2 * res_dx, init_shape[1], init_shape[2])
            labels[:, 1] -= res_dx
            labels[:, 3] -= res_dx

        if init_shape != (544, 1280, 3):
            cx = 544 / init_shape[0]
            cy = 1280 / init_shape[1]
            init_shape = (544, 1280, 3)
            labels[:, 1] *= cx
            labels[:, 3] *= cx
            labels[:, 0] *= cy
            labels[:, 2] *= cy

        if shape is not None and init_shape != shape:
            labels[:, 1] *= cx
            labels[:, 3] *= cx
            labels[:, 0] *= cy
            labels[:, 2] *= cy
        return labels


if __name__ == '__main__':
    original_path = Path("data/CW/original")
    generated_path = Path("data/CW/generated")
    real_shape = (544, 1280, 3)
    save_shape = (544 // 4, 1280 // 4, 3)
    augmented_samples = 6
    labels_json = dict()
    original_images = original_path.glob("*.jpeg")
    for image_path in original_images:
        img, labels = load_image(image_path)
        print(img, labels)
