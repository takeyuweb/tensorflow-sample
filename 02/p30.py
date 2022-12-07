import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

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

# 5回から50回に変更
model.fit(training_images, training_labels, epochs=50)
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0942 - accuracy: 0.9647
# 学習データでは 96.5%で適合

model.evaluate(test_images, test_labels)
# 313/313 [==============================] - 0s 1ms/step - loss: 0.5034 - accuracy: 0.8887
# テストデータでは 88.9% で適合

# 学習回数を5回から50回に変更した場合
# 学習データの適合率は 89.2%→96.5% と大きく改善したが
# テストデータでは 87%→89% とそれほど精度が上がっておらず剥離している
#
# 過剰適合
# 学習とテストの正解率の数値が剥離している＝学習データに過度に特化している
# これに注意する必要がある
# 回避する手法がいくつかある。今後学んでいく
