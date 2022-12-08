import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

# 入力形状を Conv2D層が期待する物に変換
# 28x28の画像60000要素→28x28x1の画像60000要素に変換
# 28x28x1 はカラーチャネル数1=モノクロの28x28。カラーなら 28x28x3 (R,G,B)
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    # 畳み込み層
    # 畳み込みの数64をランダムに初期化し、ラベルを一致させるために最も適したフィルタを学習する
    # フィルタのサイズは3x3
    # 活性化関数は正規化線形ユニット Rectified Liner Unit。負の数は0、正の数はそのまま
    # 入力形状は 28x28、1チャンネル
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(28, 28, 1)),
    # 2x2の最大値プーリング
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50)
# 1875/1875 [==============================] - 16s 9ms/step - loss: 0.0227 - accuracy: 0.9920
# Epoch 30/50

model.evaluate(test_images, test_labels)
# 畳み込み無しよりも高精度
# 313/313 [==============================] - 2s 5ms/step - loss: 1.0038 - accuracy: 0.912

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
