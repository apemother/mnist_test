import tensorflow as tf
import mnist_training


def main():
    # Load data and restore saved model
    train, test = mnist_training.load_data()
    model_path = '/home/apemother/logs/20200311-20:55:13/'
    model = tf.keras.models.load_model(model_path)

    # Make predictions and visualize
    predictions = model.predict(test)
    mnist_training.inspect_prediction_batch(test, predictions)


if __name__ == '__main__':
    main()
