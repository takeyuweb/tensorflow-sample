import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

# 正規化
# 画像は0～255の値のグレースケール
# 配列の各要素を255で割ると、0～1の間の数値になる
# TensorFlowで学習する際は正規化することで性能が向上する
# 理由は本書の範囲を超えるので説明しない、とのこと
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

# どの損失関数を使用するか？は時間を掛けて学習する必要がある、とのこと
model.compile(optimizer='adam',  # オプティマイザ：sgdオプティマイザの進化形。より効率的
              loss='sparse_categorical_crossentropy',   # 損失関数：スパースカテゴリカル交差エントロピー
              # 報告するメトリクス。学習するネットワークの正解率。入力ピクセルと出力ラベルが一致した頻度。
              metrics=['accuracy']
              )

model.fit(training_images, training_labels, epochs=5)
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.2956 - accuracy: 0.8915
# 学習データで89.2%で適合

# モデルの評価
model.evaluate(test_images, test_labels)
# 313/313 [==============================] - 0s 1ms/step - loss: 0.3415 - accuracy: 0.8772
# テストデータで 87.7%で適合

# モデル出力の調査
classifications = model.predict(test_images)
print(classifications[0])
# 10個の出力ニューロンの値
# 最後の値＝インデックス9の値＝テストデータ0番目の衣類のラベルが9である確率は88.3%であることを示す
# [3.0413913e-04 1.6315219e-07 3.5324235e-06 4.9818240e-08 3.2011350e-05
#  3.3172600e-02 6.2569656e-05 8.3425969e-02 2.6506031e-04 8.8273388e-01]
print(test_labels[0])
# テストデータ0番目のラベルを見ると9なので正解
# 9
