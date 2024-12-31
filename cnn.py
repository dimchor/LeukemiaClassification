from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

class CNN:
    def __init__(self):
        self.__nn = Sequential()

        self.__nn.add(InputLayer(shape=(128, 128, 1)));

        self.__nn.add(Conv2D(32, (3,3), activation='relu'))
        self.__nn.add(MaxPool2D((2, 2)))

        self.__nn.add(Conv2D(64, (3,3), activation='relu'))
        self.__nn.add(MaxPool2D((2, 2)))

        #3.then we ged rid of 20% of the neurons to reduce overfitting
        self.__nn.add(Dropout(0.2))
        #4. next we stretch the so far processed images into 1-d vectors
        self.__nn.add(Flatten())

        # MLP layers
        self.__nn.add(Dense(128, activation='relu'))
        self.__nn.add(Dense(64, activation='relu'))

        # Output layer
        self.__nn.add(Dense(2, activation='softmax'))

        self.__nn.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    def fit(self, dat):
        self.__nn.fit(dat, epochs=10)


    def evaluate(self, val):
        return self.__nn.evaluate(val, verbose=2)
