import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import datetime


def inspect_dataset(dataset, num):
    for example in dataset.take(num):  # Only take num examples
        image, label = example["image"], example["label"]
        plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
        plt.show()
        print("Label: %d" % label.numpy())


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    std = tf.math.reduce_std(image, axis=(0, 1))
    mean = tf.math.reduce_mean(image, axis=(0, 1))
    image = (image - mean) / std
    # Depth needs to be set to num_classes
    label = tf.one_hot(label, depth=10, dtype=np.uint8)
    # Resize the image
    # image = tf.image.resize(image, (28, 28))
    return image, label


def build_cnn(input_shape, num_classes):
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def scheduler(epoch):
    if epoch < 1:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))


def inspect_prediction_batch(dataset, predictions):
    plt.figure(figsize=(16, 16))
    i = 0
    for images, labels in dataset.take(1):
        for image, label, pred in zip(images, labels, predictions):
            if i == 36:
                break
            plt.subplot(6, 6, i + 1)
            i += 1
            image = np.squeeze(image.numpy())
            label = np.argmax(label.numpy(), axis=0)
            pred = np.argmax(pred, axis=0)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image, cmap='gray')
            plt.xlabel("Predicted Label: {}, True Label: {}".format(pred, label))
        plt.show()


def load_data():
    # Data Processing
    (raw_train, raw_test), metadata = tfds.load("mnist", as_supervised=True, with_info=True, split=['train', 'test'])
    train = raw_train.map(preprocess)
    test = raw_test.map(preprocess)
    train = train.shuffle(20000).batch(128).repeat(10)
    test = test.batch(128)
    return train, test


def main():
    train, test = load_data()

    # Build Model
    input_shape = (28, 28, 1)
    num_classes = 10
    model = build_cnn(input_shape, num_classes)
    model.summary()

    # Training
    log_dir = '/home/apemother/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(log_dir, monitor='val_loss')
    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    history = model.fit(train, validation_data=test, use_multiprocessing=True, verbose=1,
                        callbacks=[tensorboard_callback, checkpoint_callback, lr_schedule_callback],
                        validation_freq=1, steps_per_epoch=100, epochs=10, shuffle=True)

    # Evaluation
    preds = model.predict(test)
    inspect_prediction_batch(test, preds)


if __name__ == "__main__":
    main()
