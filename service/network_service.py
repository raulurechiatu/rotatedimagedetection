import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input

import util.constants as constants


def start_training(image_data, total_cards_number):
    train_data, train_labels = get_data(image_data, total_cards_number)
    X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=0)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    model = get_model()
    # model = create_ResNet50V2(constants.training_image_height, constants.training_image_width)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=5,
                        validation_data=(X_test, Y_test))

    test_loss, test_acc, test_metric2, test_metric3 = model.evaluate(X_test, Y_test, verbose=1)


def get_data(labels, total_cards_number):
    classes = np.empty((total_cards_number, 1))
    data = np.empty((total_cards_number, constants.training_image_height, constants.training_image_width), dtype='float32')
    id = 0
    for label in labels:
        for current_label in label:
            # if current_label.card_id not in current_classes:
            classes[id] = current_label.card_id
            data[id] = current_label.rectangle
            id += 1
    # print("Classes: ", classes)
    return np.reshape(data, (total_cards_number, constants.training_image_height, constants.training_image_width, 1)), classes


def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(constants.training_image_height,
                                                                                 constants.training_image_width, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(constants.number_of_classes, activation='softmax'))





    # model.add(tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, activation='relu', input_shape=(268, 182, 3)))
    # model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'))
    #
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dense(8, activation='sigmoid'))  # Final Layer using Softmax





    # model.add(tf.keras.layers.Dense(5, activation='relu', input_dim=5))
    # model.add(tf.keras.layers.Dense(2, activation='softmax'))


    return model


# Keras ResNet50V2 model
def create_ResNet50V2(width, height):
    inputs = Input(shape=(width, height, 1))

    return tf.keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
        input_tensor=inputs,
        input_shape=(width, height, 1),
        pooling=None,
        classes=constants.number_of_classes,
        classifier_activation="softmax",
    )


def gpu_setup():
    tf.get_logger().setLevel('DEBUG')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.allow_growth = True
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

