from datetime import datetime
from pathlib import Path

import tensorflow as tf
from focal_loss import BinaryFocalLoss
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import *
from tensorflow.keras.optimizers import Adam

from CWDataset import CWDataset
from datautils import *


def load_last_model():
    path = Path("model/anchor/")
    model = list(path.glob("*"))[-1]
    print("Loading model", model)
    model = tf.keras.models.load_model(str(model))
    return model


def build_anchor_model(inp_shape, anchor_counts):
    inp = Input(shape=inp_shape)
    headModel = Conv2D(32, (3, 3), activation="relu", padding = 'same')(inp)
    headModel = Conv2D(32, (3, 3), activation="relu", padding = 'same')(headModel)
    headModel = MaxPooling2D((2, 2))(headModel)

    headModel = Conv2D(64, (3, 3), activation="relu", padding = 'same')(headModel)
    headModel = Conv2D(64, (3, 3), activation="relu", padding = 'same')(headModel)
    headModel = MaxPooling2D((2, 2))(headModel)

    headModel = Conv2D(128, (3, 3), activation="relu", padding = 'same')(headModel)
    headModel = Conv2D(128, (3, 3), activation="relu", padding = 'same')(headModel)
    headModel = MaxPooling2D((2, 2))(headModel)

    headModel = Conv2D(256, (3, 3), activation="relu", padding = 'same')(headModel)
    headModel = Conv2D(256, (3, 3), activation="relu", padding = 'same')(headModel)
    headModel = MaxPooling2D((2, 2))(headModel)

    headModel = Conv2D(512, (3, 3), activation="relu", padding = 'same')(headModel)
    headModel = Conv2D(512, (3, 3), activation="relu", padding = 'same')(headModel)

    headModel = MaxPooling2D((2, 2))(headModel)

    headModel = Flatten(name="flatten")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(4 * 1024, activation="relu")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(anchor_counts[0] * anchor_counts[1], activation="sigmoid")(
        headModel
    )
    headModel = Reshape((anchor_counts[0], anchor_counts[1]))(headModel)
    model = Model(inputs=inp, outputs=headModel)
    model.compile(
        # loss="binary_crossentropy",
        loss=BinaryFocalLoss(gamma=3),  # ,  # "mean_absolute_error",
        optimizer=Adam(lr=0.001),
        metrics=["accuracy"],
    )
    return model


def prep_small_anchor(X, Y):
    return X, np.asarray(list((labels_to_markers(X[0].shape, Y[i], anchor_space // 4)) for i in range(len(Y))))


def train_anchor_model(dataset, shape, anchor_model=None):
    if anchor_model is None:
        #anchor_model = load_last_model()
        anchor_model = build_anchor_model(shape, anchors_count)
    print(anchor_model.summary())
    X, Y = prep_small_anchor(*dataset.get_train(5000, shape=shape))
    X_val, Y_val = prep_small_anchor(*dataset.get_validation(5000, shape=shape))

    show_anchors(X[0], 2, Y[0])

    print(X.shape, Y.shape)
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor="val_loss"),
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
    ]
    with tf.device("/GPU:0"):
        anchor_model.fit(
            X / 255., Y, epochs=100, validation_data=(X_val / 255., Y_val), callbacks=my_callbacks
        )
    dt_string = datetime.now().strftime("model/anchor/%d-%m-%Y %H-%M-%S")
    print("Saved " + dt_string)
    anchor_model.save(dt_string)
    return anchor_model


if __name__ == '__main__':
    real_shape = (544, 1280, 3)
    nn_shape = (544 // 4, 1280 // 4, 3)
    anchor_space = 8
    anchors_count = (1 + real_shape[0] // anchor_space, 1 + real_shape[1] // anchor_space)

    # anchor_model = build_anchor_model(nn_shape, anchors_count)

    #anchor_model = load_last_model()
   # print(anchor_model.summary())

    dataset = CWDataset(Path("data/CW/generatedBig"))

    model = train_anchor_model(dataset, nn_shape)
