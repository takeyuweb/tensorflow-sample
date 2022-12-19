import tensorflow_datasets as tfds

data, info = tfds.load("mnist", with_info=True)
print(info)
"""     features=FeaturesDict({
        'image': Image(shape=(28, 28, 1), dtype=tf.uint8),
        'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),
    }), """

# ダウンロードされたファイル
# TFRecord形式で保存される
# user@7de8226d9a68:/tensorflow-sample$ ls -al ~/tensorflow_datasets/
# downloads/        fashion_mnist/    horses_or_humans/ mnist/
# 2個のファイルにシャード化されて保存されている
# user@7de8226d9a68:/tensorflow-sample$ ls -al ~/tensorflow_datasets/mnist/3.0.1/mnist-t
# mnist-test.tfrecord-00000-of-00001   mnist-train.tfrecord-00000-of-00001
