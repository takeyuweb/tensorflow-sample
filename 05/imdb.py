import tensorflow as tf
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
import string

stopwords = ["a", "the", "yourselves"]

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
for item in train_data:
    sentence = str(item['text'].decode('UTF-8').lower())
    soup = BeautifulSoup(sentence)
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        filtered_sentence = filtered_sentence + word + " "
    imdb_sentences.append(filtered_sentence)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)

print(tokenizer.word_index)
