import numpy as py #handling numberical operations
import matplotlib.pyplot as plt #for visualizaiton
from sklearn.model_selection import train_test_split #splitting into training/testing sets
from sklearn.neighbors import KNeighborsClassifier # the KNN model
#neighbors == close points in data
from sklearn.datasets import load_iris #iris dataset

data = load_iris() ##loading the Iris dataset into a varibale called data

X = data.data #Features: The measurements of the flowers
y = data.target #Labels: the species of the flower

#X == storing the features
#in machine learning


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#random_state=42 setting this makes sure our data is split the same way each time we run this code, which is helpful for consistency. 

knn = KNeighborsClassifier(n_neighbors=3) #Creating a KNN model with 3 neighbors

knn.fit(X_train, y_train) #training the model with our training data

#knn.fit(): method == trains the model on the data we provide

accuracy = knn.score(X_test, y_test) #Testing the model and getting the accuracy

print("Model accuracy:", accuracy)

sample = [[5.1, 3.5, 1.4, 0.2]] #A new sample with measurements to classify

prediction = knn.predict(sample)

print("Predicted species:", prediction)
