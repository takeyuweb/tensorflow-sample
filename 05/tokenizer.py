import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

tokenizer = Tokenizer(num_words=100)    # トークンの最大数を指定してインスタンス化
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)
# [[1, 2, 3, 4, 5], [1, 2, 3, 6, 5], [2, 7, 4, 1]]
