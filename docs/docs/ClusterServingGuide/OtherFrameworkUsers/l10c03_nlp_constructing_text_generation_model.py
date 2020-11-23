# Related url: https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l10c03_nlp_constructing_text_generation_model.ipynb
# Generating some new lyrics from the trained model

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Other imports for processing data
import string
import numpy as np
import pandas as pd

# DATA PREPROCESSING
# First to get the dataset of the Song Lyrics dataset on Kaggle by:
# !wget --no-check-certificate \
#    https://drive.google.com/uc?id=1LiJFZd41ofrWoBtW-pMYsfz1w8Ny0Bj8 \
#    -O /tmp/songdata.csv

# Then to generate a tokenizer with the songdata.csv
def tokenize_corpus(corpus, num_words=-1):
  # Fit a Tokenizer on the corpus
  if num_words > -1:
    tokenizer = Tokenizer(num_words=num_words)
  else:
    tokenizer = Tokenizer()
  tokenizer.fit_on_texts(corpus)
  return tokenizer

def create_lyrics_corpus(dataset, field):
  # Remove all other punctuation
  dataset[field] = dataset[field].str.replace('[{}]'.format(string.punctuation), '')
  # Make it lowercase
  dataset[field] = dataset[field].str.lower()
  # Make it one long string to split by line
  lyrics = dataset[field].str.cat()
  corpus = lyrics.split('\n')
  # Remove any trailing whitespace
  for l in range(len(corpus)):
    corpus[l] = corpus[l].rstrip()
  # Remove any empty lines
  corpus = [l for l in corpus if l != '']

  return corpus

# Read the dataset from csv
dataset = pd.read_csv('/tmp/songdata.csv', dtype=str)
# Create the corpus using the 'text' column containing lyrics
corpus = create_lyrics_corpus(dataset, 'text')
# Tokenize the corpus
tokenizer = tokenize_corpus(corpus)

# Get the uniform input length (max_sequence_len) of the model
max_sequence_len=0
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    max_sequence_len=max(max_sequence_len,len(token_list))

# Load the saved model which is trained on the Song Lyrics dataset
model=tf.keras.models.load_model("path/to/model")

# Generate new lyrics with some "seed text"
seed_text = "im feeling chills" # seed text can be customerized
next_words = 100    # this defined the length of the new lyrics

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]   # convert the seed text to ndarray
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')    # pad the input for equal length
    predicted = np.argmax(model.predict(token_list), axis=-1)   # get the predicted word index
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word  # add the predicted word to the seed text
print(seed_text)
