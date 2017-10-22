import pandas as pd
import numpy as np
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

file = 'data.csv'
headerNames = ['elapsed', 'blink_length', 'drowsy']

dataset = pd.read_csv(file, sep=',', header=None, names=headerNames)
dataset.keys()

X = dataset.loc[:, 'elapsed':'blink_length']
y = dataset.loc[:, 'drowsy']
categorical_labels = to_categorical(y, num_classes=None)

X_train, X_test, y_train, y_test = train_test_split(X.values, categorical_labels, test_size=0.40, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
cb = [ks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=0, mode='auto')]

model = Sequential()
model.add(Dense(156, input_dim=2, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, callbacks=cb, batch_size=128, validation_data=(X_test,y_test))
score = model.evaluate(X_test, y_test,verbose=1)

model.save('drowsy.h5')

print(score)