# https://github.com/lmoroney/tfbook/blob/master/chapter3/Horse_or_Human_NoValidation.ipynb
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

training_dir = '03/horse-or-human/training/'
validation_dir = '03/horse-or-human/validation/'

# 学習データの管理
train_datagen = ImageDataGenerator(rescale=1/255)
# ディレクトリ構造を利用して自動的にラベル付けする
train_generator = train_datagen.flow_from_directory(
    training_dir,  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=128,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# 検証データの管理
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'
)

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
history = model.fit(train_generator, epochs=15,
                    validation_data=validation_generator)

# Epoch 15/15
# 9/9 [==============================] - 4s 452ms/step - loss: 0.0278 - accuracy: 0.9903 - val_loss: 1.9847 - val_accuracy: 0.8320
#
# 学習データによる適合率は99%だが、検証用データでは83%にとどまる
# モデルが過剰適合している
