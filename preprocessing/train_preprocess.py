"""
Train convolutional network for sentiment analysis. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf
""
Original taken from: https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/trainGraph.py
""
"""

import numpy as np
import data_helpers
from w2v import train_word2vec
import csv

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D

np.random.seed(2)

# Parameters
# ==================================================
#
# Model Variations. See Kim Yoon's Convolutional Neural Networks for 
# Sentence Classification, Section 3 for detail.

model_variation = 'CNN-static' #  CNN-rand | CNN-non-static | CNN-static
print('Model variation is %s' % model_variation)

# Model Hyperparameters
sequence_length = 56
embedding_dim = 128          
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

# Training parameters
batch_size = 32
num_epochs = 100
val_split = 0.1

# Word2Vec parameters, see train_word2vec
min_word_count = 1  # Minimum word count                        
context = 10        # Context window size    

# Data Preparatopn
# ==================================================
#
# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data()
print("Vocabulary Size: {:d}".format(len(vocabulary)))
np.savetxt('data.txt', x) 
np.savetxt('label.txt', y) 

with open('vocab.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in vocabulary.items():
        writer.writerow([key, value])
csv_file.close()

if model_variation=='CNN-non-static' or model_variation=='CNN-static':
    embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context) 
    if model_variation=='CNN-static':
        x = embedding_weights[0][x]
elif model_variation=='CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')    

embedding = np.asarray(embedding_weights)
new_embeding = embedding.reshape(21323*128,1)
np.savetxt("embedding.txt", np.array(new_embeding))

