# https://github.com/lmoroney/tfbook/blob/master/chapter3/Horse_or_Human_NoValidation.ipynb
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import RMSprop

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

# ETLパターン
# Extrac Transform Load というプロセスで構成することで
# データやスキーマの変更に影響をうけにくいデータパイプラインを構成できる
# 1台のマシンでも複数台のクラスタ構成でも同じ基本構造が使用できる

# Extractここから
# データの置き場に関わらず Transform 可能なようにデータを準備する
data = tfds.load('horses_or_humans', split='train', as_supervised=True)
val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)
# Extractここまで

# Transformここから

# 画像拡張
# マッピング関数を作成して処理する


def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255)
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, 40, interpolation="NEAREST")
    return image, label


# バッチ処理
# シャッフルしたものを10個ずつ取り出してバッチ処理する
# バッチ処理することでより効率的に学習できるようになる
# CPUを使うTransformが終わったらGPUを使うLoadを進める
# Load中CPUは暇になるので、次のバッチのTransformを進める
# Transform中でGPUが暇になったら、前のバッチのLoadを進める
# このようにして効率的にコンピューティング資源を使える
train_batches = data.map(augmentimages).shuffle(100).batch(10)

val_batches = val_data.batch(32)

# Transform ここまで

# Load ここから
# 学習のためにデータをニューラルネットワークに読み込むこと
# 学習実行
history = model.fit(train_batches, epochs=10,
                    validation_data=val_batches, validation_steps=1)
# Load ここまで
