# https://github.com/lmoroney/tfbook/blob/master/chapter3/Horse_or_Human_NoValidation.ipynb
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import RMSprop

data = tfds.load('horses_or_humans', split='train', as_supervised=True)
val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)

# バッチ処理
# シャッフルしたものを10個ずつ取り出してバッチ処理する
# バッチ処理することでより効率的に学習できるようになる
train_batches = data.shuffle(100).batch(10)

val_batches = val_data.batch(32)

model = tf.keras.models.Sequential([
    # 3x3のフィルタを16個、入力は300x300のRGB
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(300, 300, 3)),
    # 半分のサイズに
    tf.keras.layers.MaxPooling2D(2, 2),
    # 画像が大きいので、それぞれの特徴を強調した多くの小さな画像を生成するために層を重ねる
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ネットワークの学習
# 損失関数とオプティマイザをコンパイルする
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
# 学習実行
history = model.fit(train_batches, epochs=10,
                    validation_data=val_batches, validation_steps=1)
