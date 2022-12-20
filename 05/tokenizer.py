import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]

# OOV=Out-Of-Vocabulary
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)
print(padded)
# [[ 0  0  0  2  3  4  5  6]
#  [ 0  0  0  2  3  4  7  6]
#  [ 0  0  0  0  3  8  5  2]
#  [ 9 10 11 12 13 14 15  2]]

text_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(text_data)

print(word_index)
# {'<OOV>': 1, 'today': 2, 'is': 3, 'a': 4, 'sunny': 5, 'day': 6, 'rainy': 7, 'it': 8}
print(sequences)
# [[2, 3, 4, 1, 6], [1, 8, 1, 7, 1]]
# today is a <OOV> day
# <OOV> itt <OOV> rainy <OOV>
# 前者は本来の意味に近くなった
