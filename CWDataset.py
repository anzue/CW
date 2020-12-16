import json
from random import random, shuffle
from ArrayLoader import ArrayLoader
from datautils import *


class CWDataset:
    def __init__(self, path: Path, train_percent=0.8):
        img_list = [x for x in path.glob("**/*.jpeg")]
        # shuffle(img_list)
        self.train_img = []
        self.test_img = []
        self.validation_img = []

        for i in range(len(img_list)):
            if random() <= train_percent:
                self.train_img.append(img_list[i])
            elif len(self.validation_img) < len(self.test_img):
                self.validation_img.append(img_list[i])
            else:
                self.test_img.append(img_list[i])

        print("Train", len(self.train_img))
        print("Test ", len(self.test_img))
        print("Vali ", len(self.validation_img))
        self.train = ArrayLoader(self.train_img)
        self.test = ArrayLoader(self.test_img)
        self.validation = ArrayLoader(self.validation_img)
        with open(str(path.joinpath("labels.json"))) as f:
            self.labels = json.load(f)

    def get_labels(self, image):
        return np.asarray(self.labels[image]["labels"])

    def get_scaled_labels(self, image, dx, dy):
        arr = self.labels[image]["labels"]
        return np.asarray(list((arr[i][0] * dx,
                                arr[i][1] * dy,
                                arr[i][2] * dx, arr[i][3] * dy) for i in range(len(arr))))

    def get(self, data, count, shape):
        real_count = min(count, data.size())
        imgs, names = data.get(range(real_count), with_names=True, shape=shape)
        if shape is not None:
            return imgs, list(
                (self.get_scaled_labels(str(name.name), shape[1] / 1280., shape[0] / 544., )) for name in names)
        else:
            return imgs, list((self.get_labels(str(name.name))) for name in names)

    def get_train(self, count, shape=None):
        return self.get(self.train, count, shape)

    def get_test(self, count, shape=None):
        return self.get(self.test, count, shape)

    def get_validation(self, count, shape=None):
        return self.get(self.validation, count, shape)
