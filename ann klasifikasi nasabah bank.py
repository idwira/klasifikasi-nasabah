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

# Membuat tiga dummy variables untuk tiga country
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Membuang satu dummy variable country di index 0, cukup dua
# Menghindari jebakan dummy variable
# Update range X, adalah index 1 hingga index terakhir di kolom terakhir
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

# Mengimport Keras library and modul-modul nya dg backend TensorFlow
# Keras akan membangun Deep Neural Network berdasarkan TensorFlow
# Modul Sequential untuk initialize neural network nya
# Modul Dense untuk membangun layers dari ANN 
import keras
from keras.models import Sequential
from keras.layers import Dense

# Memulai proses ANN
# classifier = Sequential()

""" Dense func melakukan inisialisasi bobot tiap node secara random ke angka kecil
relu (rectified linear units) untuk hidden layers
sigmoid function untuk output layer

Menambah input layer dan hidden layer pertama:
Buat 6 node di hidden layer, hasil rata-rata jumlah input layer dan output layer
11 input layer (independent variable) + 1 output layer (biner dependent variabel) """
# classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))

# Menambah hidden layer kedua
# classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))

# Menambah output layer
# classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

""" Mengkompile ANN:
Input dari optimizer parameter, 'adam' salah satu algo. stochastic gradient descent
binary_crossentropy optimized loss function, utk optimal weights
metrics, criterion yang dipilih untuk mengevaluasi model
accuracy criterion, improve akurasi saat weights di update tiap batch (saat fit ann) """
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

""" Fitting ANN ke Training set:
Angka epochs, jumlah training ann pada seluruh training set
Akurasi meningkat pada tiap putaran (tiap epochs) 
Ukuran batch, update bobot setiap satu bacth observasi, disini tiap 10 data
4 argumen; matrix of features, dependent var. vector, batch size, jumlah epoch """
# classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

""" Bagian 3 - Membangun prediksi dan mengevaluasi model
Prediksi hasil test set
jika y_pred > 0.5 maka True """
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)

# Membangun Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

""" K-fold Cross Validation
Untuk akurasi yang lebih relevan di test set
K buah kombinasi training dan test set
Menghitung rata-rata akurasi nya, juga standar deviasi nya utk mendapatkan variance

Problem, model dibangun oleh Keras, sementara fungsi K-Fold Val. adalah Scikit-learn
Mengkombinasikan Keras dan Scikit-Learn dengan modul 'Keras wrapper' dari Keras
yang akan wrap K-fold CV dari Scikit-learn menjadi Keras Model
Dengan kata lain, include K-fold CV ke dalam classifier keras kita """	

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
	classifier = Sequential()
	classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
	classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
	classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
	
#Hitung mean dan variance
mean = accuracies.mean()
variance = accuracies.std() 
