# Klasifikasi Nasabah Dengan Artificial Neural Network
# https://github.com/idwira/klasifikasi-nasabah

# Bagian 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Sudah dipilih independent variable index 3 sampai dengan 12
# X independent variable
# Y dependent variable
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Pre-process Data Dimulai
# Encoding categorical data format string menjadi number value
# X_1 Encode France, Germany dan Spain menjadi 0, 1 dan 2
# X_2 Encode Female menjadi 0 dan Male menjadi 1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Membuat tiga dummy variables untuk country
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Membuang satu dummy variable country di index 0, cukup dua
# Menghindari jebakan dummy variable
# Update X, adalah index 1 index terakhir di kolom terakhir
X = X[:, 1:]

# Membagi dataset ke dalam set Training set dan set Test
# Train 8000 data, observasi test 2000 data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# mengatur komputasi lebih efisien
# scaled X_train dan X_test
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Akhir Pre-process data

# Bagian 2 - Membangun ANN

# Mengimport Keras libraries and packages nya dg backend tensorflow
# Sequential untuk initialize neural network nya
# Dense untuk membangun layers dari ANN 
import keras
from keras.models import Sequential
from keras.layers import Dense

# Memulai proses ANN
classifier = Sequential()

# Menambah input layer dan hidden layer pertama
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Menambah hidden layer kedua
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Menambah output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Mengkompile ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting ANN ke Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Bagian 3 - Membangun prediksi dan mengevaluasi model

# Prediksi hasil test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Membangun Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)