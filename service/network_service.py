import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import util.constants as constants


def predict(model, image):
    return model.predict(image), (np.where(model.predict(image)>0.9)[1][0], model.predict(image).max())


def start_training(image_data, total_cards_number):
    model = create_ResNet50V2(constants.training_image_height, constants.training_image_width)

    if constants.retrain:
       model = train_model(image_data, total_cards_number, model)
    else:
        model.load_weights("resources/models/" + constants.weights_to_load)
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

    return model


def train_model(image_data, total_cards_number, model):
    train_data, train_labels = get_data(image_data, total_cards_number)
    X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=0)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    # (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    # X_train, X_test = X_train / 255.0, X_test / 255.0

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=10, batch_size=8,
                        validation_data=(X_test, Y_test))

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
    print("Loss: ", test_loss, " Acc: ", test_acc)

    model.save('resources/models/' + constants.weights_file_name)

    return model


def get_data(labels, total_cards_number):
    classes = np.empty((total_cards_number, 1), dtype='float16')
    data = np.empty((total_cards_number, constants.training_image_height, constants.training_image_width), dtype='float32')
    id = 0
    for label in labels:
        for current_label in label:
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
    # inputs = Input(shape=(width, height, 1))

    return tf.keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
        # input_tensor=inputs,
        input_shape=(width, height, 1),
        pooling=None,
        classes=constants.number_of_classes,
        classifier_activation="softmax",
    )


def gpu_setup():
    # This line stops the usage of GPU in tensorflow, needed because the network would not start on GPU
    # tf.config.set_visible_devices([], 'GPU')

    tf.get_logger().setLevel('DEBUG')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            # tf.config.allow_growth = True
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=6000)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)


def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
