#CMPSC497 Midterm Project
#Damien Kula and Hamid Shah
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api

#Opening following dataset: https://www.kaggle.com/datasets/sameedatif/tone-analysis on Kaggle
#I use an outside tool to automatically shuffle the data and allow for randomness to get every label
dataset_path = "" #Input your filepath here directing to tone_dataset_shuffled.txt

dataset = open(dataset_path, encoding="utf8")
training_data = []
training_labels = []
testing_data = []
testing_labels = []

#The data has 3356 sentences so we use ~80% for training
for i in range(2680):
    sentence = dataset.readline().strip()
    values = sentence.split(" || ")
    label = values[1][:-1]
    training_data.append(values[0])
    training_labels.append(label)

for i in range(676):
    sentence = dataset.readline().strip()
    if not sentence: #in case there is no sentence it wont add extra empty strings
        break
    values = sentence.split(" || ")
    label = values[1][:-1]
    testing_data.append(values[0])
    testing_labels.append(label)

dataset.close()

# Initial Tokenization
max_words = 5000
max_length = 200
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(training_data)
sequences = tokenizer.texts_to_sequences(training_data)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

test_sequences = tokenizer.texts_to_sequences(testing_data)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Convert labels to numerical values
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(training_labels)
label_sequences = np.array(label_tokenizer.texts_to_sequences(training_labels))
label_sequences = label_sequences.reshape(-1)

test_label_sequences = np.array(label_tokenizer.texts_to_sequences(testing_labels))
test_label_sequences = test_label_sequences.reshape(-1)

#Word2Vec embeddings produced from model
w2v_model = Word2Vec(sentences=[sentence.split() for sentence in training_data], vector_size=100, window=5, min_count=1, workers=4)
word_vectors = w2v_model.wv

#GloVe embeddings
glove_model = api.load("glove-wiki-gigaword-100")
embedding_matrix = np.zeros((max_words, 100))

for word, i in tokenizer.word_index.items():
    if i < max_words:
        w2v_vector = word_vectors[word] if word in word_vectors else None
        glove_vector = glove_model[word] if word in glove_model else None
        
        #This combines the two embeddings by averaging them up
        if w2v_vector is not None and glove_vector is not None:
            embedding_matrix[i] = (w2v_vector + glove_vector) / 2 
        elif w2v_vector is not None:
            embedding_matrix[i] = w2v_vector
        elif glove_vector is not None:
            embedding_matrix[i] = glove_vector

# Define CNN model with combined embeddings
model = keras.Sequential([
    keras.layers.Embedding(input_dim=max_words, output_dim=100, input_length=max_length, weights=[embedding_matrix], trainable=False),
    keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(label_tokenizer.word_index) + 1, activation='softmax')
])

# Compile utilizing adam
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(padded_sequences, label_sequences, epochs=25, batch_size=32, validation_split=0.2)

# Evaluate model
test_loss, test_acc = model.evaluate(padded_test_sequences, test_label_sequences)
print(f"Test Accuracy: {test_acc}")

# Make predictions
predictions = model.predict(padded_test_sequences)
predicted_labels = np.argmax(predictions, axis=1)