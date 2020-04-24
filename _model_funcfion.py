import tensorflow as tf
from tensorflow.keras import models, layers

# 1 분산
x = layers.Input(shape=(3,))
h = layers.Activation('relu')(layers.Dense(2)(x))
y = layers.Activation('softmax')(layers.Dense(1)(h))
model = models.Model(x, y)
model.compile(loss='categorical_crossentropy' ,optimizer='adam', metrics=['accuracy'])


# 2 연쇄
model = models.Sequential()
model.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
model.add(layers.Dense(Nout, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

