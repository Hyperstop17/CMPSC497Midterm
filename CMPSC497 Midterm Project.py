#CMPSC497 Midterm Project
#Damien Kula and Hamid Shah
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

#Opening following dataset: https://www.kaggle.com/datasets/sameedatif/tone-analysis on Kaggle
dataset = open("C:\\Users\\damie\\Documents\\Python\\Datasets\\tone_dataset_shuffled.txt", encoding="utf8")

training_data = []
training_labels = []
testing_data = []
testing_labels = []

#the data has 3356 sentences so we use ~70% for training
for i in range(2410):
    sentence = dataset.readline()
    values = sentence.split(" || ")
    label = values[1][:-1]
    training_data.append(values[0])
    training_labels.append(label)

# Remaining ~30% for testing
for i in range(946):
    sentence = dataset.readline()
    if not sentence: #Avoids possible padding from overestimation
        break
    values = sentence.split(" || ")
    label = values[1][:-1]
    testing_data.append(values[0])
    testing_labels.append(label)

dataset.close()