import tensorflow as tf
import tensorflow_datasets as tfds


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = tfds.as_numpy(
    tfds.load(
        'fashion_mnist',
        # traint と test の画像とラベルの入ったデータセットアダプタの配列を返す
        split=['train[:20%]', 'test'],  # train の先頭から20%のデータを使う
        batch_size=-1,  # 全件
        as_supervised=True  # タプルで返す
    )
)

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])
