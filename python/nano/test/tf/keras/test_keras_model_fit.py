from bigdl.nano.tf.keras import Model
import tensorflow as tf
import pathlib


class CustomModel(Model):
    def __init__(self, custom_layers: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_layers = custom_layers

    def call(self, x):
        for layer in self.custom_layers:
            x = layer(x)
        return x


def test_fit_batch_size():

    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    batch_size = 32
    img_height = 180
    img_width = 180
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width))

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width))
    class_names = train_ds.class_names

    # Create Model
    # inputs = tf.keras.Input(shape=(784,), name="digits")
    # x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    # x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    # outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)
    # model = Model(inputs=inputs, outputs=outputs)

    num_classes = len(class_names)
    model = CustomModel(custom_layers=[
        tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    # Prepare data
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data (these are NumPy arrays)

    # Reserve 10,000 samples for validation
    # x_val = x_train[-10000:]
    # y_val = y_train[-10000:]
    # x_train = x_train[:-10000]
    # y_train = y_train[:-10000]

    # Complie Model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # "Fit model on training data"

    history = model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=1,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=val_ds,
        perf_tune=True
    )


if __name__ == '__main__':
    test_fit_batch_size()
