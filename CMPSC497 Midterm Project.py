#CMPSC497 Midterm Project
#Damien Kula and Hamid Shah

#Opening following dataset: https://www.kaggle.com/datasets/sameedatif/tone-analysis on Kaggle
dataset = open("C:\\Users\\damie\\Documents\\Python\\Datasets\\tone_dataset_shuffled.txt", encoding="utf8")

training_data = []
training_labels = []

#the data has 3356 sentences so we use ~70% for training
for i in range(2410):
    sentence = dataset.readline()
    values = sentence.split(" || ")
    label = values[1][:-1]
    training_data.append(values[0])
    training_labels.append(label)

print(training_data)
print(training_labels)
dataset.close()