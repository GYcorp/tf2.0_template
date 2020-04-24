import tensorflow as tf
from tensorflow.keras import models, layers

# 1 분산
class ANN_models_class(models.Model):
    def __init__(self, Nin, Nh, Nout):
        # Prepare network layers and activate functions
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')
        # Connect network elements
        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))
        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy'])
    
# 2 연쇄
class ANN_seq_class(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
        super().__init__()
        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
        self.add(layers.Dense(Nout, activation='softmax'))
        
        self.compile(loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy'])
    
    