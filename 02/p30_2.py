import tensorflow as tf

# 学習の停止
# 目的の適合率に達した時点で打ち切る


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# コールバックで目的の適合率で中断できるようにする
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])
# 1872/1875 [============================>.] - ETA: 0s - loss: 0.1300 - accuracy: 0.9508
# Reached 95% accuracy so cancelling training!
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.1302 - accuracy: 0.9507
