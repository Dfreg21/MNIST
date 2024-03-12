import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn import metrics

(xtrain, ytrain), (xtest, ytest) = fashion_mnist.load_data()
print(ytrain)

number = np.random.randint(0,60000);

plt.imshow(xtrain[number],cmap=plt.cm.gray)
plt.title('numer = ' + str(ytrain[number]))
plt.show()

xtrain = xtrain.astype('float32')/255
xtest = xtest.astype('float32')/255

print(np.shape(xtrain))

xtrain = xtrain.reshape(xtrain.shape[0], 28, 28, 1)
xtest = xtest.reshape(xtest.shape[0], 28, 28, 1)

ytrain = keras.utils.to_categorical(ytrain, 10)
ytest = keras.utils.to_categorical(ytest, 10)

model = Sequential()

# Añadimos la primera capa
model.add(Conv2D(25,kernel_size=(12,12),strides=(1,1),padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

# Añadimos la segunda capa
model.add(Conv2D(25,kernel_size=(12,12),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Hacemos un flatten 
model.add(Flatten())
model.add(Dense(200,activation='relu'))
model.add(Dense(200,activation='relu'))

# Añadimos una capa softmax 
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

hist = model.fit(xtrain,ytrain,batch_size=64,epochs=12,verbose=1,validation_data=(xtest,ytest))

# Precisión
plt.figure()
plt.grid()
plt.plot(hist.history['accuracy'],lw=2)
plt.plot(hist.history['val_accuracy'],lw=2)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Pérdida
plt.figure()
plt.grid()
plt.plot(hist.history['loss'],lw=2)
plt.plot(hist.history['val_loss'],lw=2)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

ypred = model.predict(xtest)

y_pred = np.argmax(ypred,axis=1)
y_test = np.argmax(ytest,axis=1)

# Matriz de Confusión
print('Confusion matrix: \n', metrics.confusion_matrix(y_test,y_pred))

model.save('CNN_FMNIST.h5')
tf.keras.backend.clear_session()
