import os
import stat
from copy import deepcopy
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from label import Label


def convert_wepb(path: Path):
    webpList = [x for x in path.glob("**/*.webp")]
    print("Found", len(webpList), "webp files. Converting")
    for x in webpList:
        pathStr = str(Path.cwd().joinpath(x))
        im = Image.open(pathStr).convert("RGB")
        im.save(str(x.with_suffix(".jpeg")), "jpeg")
        os.chmod(str(x), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        x.unlink()
        print("saved ", x, " as ", x.with_suffix(".jpeg"))


# show image with rects
def draw_labels(image, labels=None):
    draw_img = Image.fromarray(deepcopy(image))
    draw = ImageDraw.Draw(draw_img)
    for label in labels:
        tmp = Label(label).display_form()
        prev = tmp[-1]
        for point in tmp:
            draw.line((prev[0], prev[1], point[0], point[1]), fill=128, width=10)
            prev = point
    return draw_img


def convert_predictions_to_seq(pred):
    res = []
    for x in pred:
        res.append(Label(x).pascal_voc())
    return res


def plot_images(source, limit, *labels_seq):
    for i, img in enumerate(source):
        if i >= limit:
            break
        for j, labels in enumerate(labels_seq):
            plot = plt.subplot(1, len(labels_seq), 1 + j)
            if (labels is not None) and (len(labels) > i):
                lab = labels[i]
            else:
                lab = ()
            plot.imshow(draw_labels(img, lab))
        plt.show()


def rect_area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def rect_intersection(rect1, rect2):
    return max(min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]), 0) * max(
        min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]), 0)


def iou(rect1, rect2):
    a1 = rect_area(rect1)
    a2 = rect_area(rect2)
    ainters = rect_intersection(rect1, rect2)
    return ainters / (a1 + a2 - ainters)


def labels_to_bitmap(labels, res_shape, source_shape):
    image = np.array(np.zeros(res_shape, dtype=float))
    for label in labels:
        tmp = [int(label[0] * res_shape[1] / source_shape[1]),
               int(label[1] * res_shape[0] / source_shape[0]),
               int(label[2] * res_shape[1] / source_shape[1]),
               int(label[3] * res_shape[0] / source_shape[0]),
               ]

        image[tmp[1]: tmp[3], tmp[0]:tmp[2]] = 1

    return image


def prepare_dataset(images, labels, input_shape):
    X_raw = []
    Y_raw = []
    for i in tqdm(range(len(images))):
        if images[i].shape[:2] != input_shape[:2]:
            arr = cv2.resize(
                deepcopy(images[i]),
                dsize=(input_shape[1], input_shape[0]),
                interpolation=cv2.INTER_AREA,
            ) / 255.0
        else:
            arr = deepcopy(images[i]) / 255.0

        X_raw.append(arr)
        Y_raw.append(labels_to_bitmap(labels[i], input_shape, images[i].shape))

    X = np.asarray(X_raw)
    Y = np.asarray(Y_raw)
    del X_raw
    del Y_raw

    return X, Y


def accuracy(original, prediction, threshold=0.7):
    predicted = 0
    total = 0
    for img_id in range(len(original)):
        total += len(original[img_id])
        for l1 in original[img_id]:
            for l2 in prediction[img_id]:
                if iou(l1, l2) >= threshold:
                    predicted += 1
                    break

    return predicted / total


def probs_to_labels_with_coefs(probs, threshold, cx, cy):
    from scipy.ndimage.measurements import label

    cpy = deepcopy(probs)
    cpy[cpy >= threshold] = 1
    cpy[cpy < threshold] = 0
    cpy = cpy.astype(int)
    labeled, ncomponents = label(cpy, np.ones((3, 3), dtype=np.int))

    boxes = []
    for i in range(1, ncomponents + 1):
        indices = np.where(labeled == i)
        if len(indices[0]) > 5:  # todo cehck this
            y1 = indices[0].min() * cy
            y2 = indices[0].max() * cy
            x1 = indices[1].min() * cx
            x2 = indices[1].max() * cx
            boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes)


def probs_to_labels(probs, threshold, coe):
    return probs_to_labels_with_coefs(probs, threshold, coe, coe)


def apply_heatmap(heatmap, level=0.7):
    #     print(heatmap.max())
    #     heatmap[heatmap < level] = 0
    #     heatmap[heatmap >= level] = 1

    #     kernel = np.ones((5, 5), np.uint8)
    #     heatmap = cv2.erode(heatmap, kernel, iterations=1)

    return heatmap


def show_on_data(img, labels, coun, model, shape=(544, 1080), threshold=0.7):
    for i in range(coun):
        plot_images((img[i],), 1, (labels[i],))
        x_tmp, y_tmp = prepare_dataset(
            img[i: i + 1],
            (labels[i],),
            shape,
        )
        predict = model.predict(x_tmp)
        real_heatmap = labels_to_bitmap(labels[i], shape, img[i].shape)
        plt1 = plt.subplot(1, 2, 1)
        plt1.imshow(np.abs(real_heatmap), cmap="gray")
        plt2 = plt.subplot(1, 2, 2)
        plt2.imshow(np.abs(apply_heatmap(predict[0], threshold)), cmap="gray")
        plt.show()


def create_video(model, source, result,
                 dataset_transaformer, result_transformer,
                 start_pos, total_count, fps):
    arr = []
    vidcap = cv2.VideoCapture(source)
    print(vidcap)
    print(vidcap.get(cv2.CAP_PROP_FOURCC))
    success, image = vidcap.read()
    count = 0
    i = 0

    while success:
        if len(arr) > total_count:
            break
        i += 1
        image = image[(720 - 544) // 2: (720 + 544) // 2, :, :]
        if i > start_pos:
            arr.append(image)
        # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

    arr = np.asarray(arr)

    X = dataset_transaformer(arr)
    predictions = model.predict(X)
    Y_pred = []
    for pred in tqdm(predictions):
        Y_pred.append(result_transformer(pred))

    video_file = result
    video = cv2.VideoWriter(
        video_file,
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps,
        (arr.shape[2], arr.shape[1]),
    )

    for i in range(arr.shape[0]):
        arr[i] = draw_labels(arr[i], Y_pred[i])
        video.write(arr[i])

    video.release()
    print(arr.shape)


def labels_to_markers(img_shape, labels, space):
    res = np.zeros((img_shape[0] // space + 1, img_shape[1] // space + 1))
    for label in labels:
        # print(label)
        res[
        int(label[1]) // space: int(label[3]) // space,
        int(label[0]) // space: int(label[2]) // space,
        ] = 1

    return np.asarray(res)


def anchors_to_labels(anchors, anchor_space, threshold):
    from scipy.ndimage.measurements import label

    cpy = deepcopy(anchors)
    cpy[cpy >= threshold] = 1
    cpy[cpy < threshold] = 0
    cpy = cpy.astype(int)
    labeled, ncomponents = label(cpy, np.ones((3, 3), dtype=np.int))

    boxes = []
    for i in range(1, ncomponents + 1):
        indices = np.where(labeled == i)
        if len(indices[0]) > 2:  # todo cehck this
            y1 = indices[0].min() * anchor_space
            y2 = indices[0].max() * anchor_space
            x1 = indices[1].min() * anchor_space
            x2 = indices[1].max() * anchor_space
            boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes)


def show_anchors(img, space, *markers):
    threshold = 0.6

    plt1 = plt.subplot(1, 1 + len(markers), 1)
    plt.imshow(img)

    for ii, marker in enumerate(markers):
        to_show = deepcopy(img)
        for i in range(marker.shape[0]):
            for j in range(marker.shape[1]):
                color = (int(255 * marker[i][j]), int(255 * (1 - marker[i][j])), 0)

                to_show = cv2.rectangle(
                    to_show,
                    (j * space - 2, i * space - 2),
                    (j * space + 2, i * space + 2),
                    color,
                )
        plt2 = plt.subplot(1, 1 + len(markers), ii + 2)
        plt.imshow(to_show)
    plt.show()


def show_label_anchors(img, labels, space):
    show_anchors(img, space, labels_to_markers(img.shape, labels, space))


def prepare_anchor_dataset(arrays, labels, shape, anchor_space):
    X = []
    for i in range(len(arrays)):
        X.append(cv2.resize(
            deepcopy(arrays[i]),
            dsize=(shape[1], shape[0]),
            interpolation=cv2.INTER_AREA,
        ))
    X = np.asarray(X) / 255.
    Y_raw = []
    for i in tqdm(range(len(X))):
        Y_raw.append(labels_to_markers(arrays[i].shape, labels[i], anchor_space))

    Y = np.asarray(Y_raw)
    del Y_raw
    del arrays
    del labels
    print(X.shape)
    print(Y.shape)
    return X, Y


def anchors_to_labels(anchors, anchor_space, threshold):
    from scipy.ndimage.measurements import label

    cpy = deepcopy(anchors)
    cpy[cpy >= threshold] = 1
    cpy[cpy < threshold] = 0
    cpy = cpy.astype(int)
    labeled, ncomponents = label(cpy, np.ones((3, 3), dtype=np.int))

    boxes = []
    for i in range(1, ncomponents + 1):
        indices = np.where(labeled == i)
        if len(indices[0]) > 2:  # todo cehck this
            y1 = indices[0].min() * anchor_space
            y2 = indices[0].max() * anchor_space
            x1 = indices[1].min() * anchor_space
            x2 = indices[1].max() * anchor_space
            boxes.append([x1, y1, x2, y2])

    res = np.asarray(boxes)
    del boxes
    return res
