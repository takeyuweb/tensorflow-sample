import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

tokenizer = Tokenizer(num_words=100)    # トークンの最大数を指定してインスタンス化
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
