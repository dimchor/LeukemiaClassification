from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, InputLayer, MaxPool2D, Flatten, Dropout, BatchNormalization
import time, json

class CNN:
    def __init__(self, shape, **kwargs):
        self.__nn = Sequential()

        self.__nn.add(InputLayer(shape=shape));

        self.__nn.add(Conv2D(32, (3,3), activation='relu'))
        self.__nn.add(BatchNormalization())
        self.__nn.add(MaxPool2D((2, 2)))
        self.__nn.add(Dropout(0.5))

        self.__nn.add(Conv2D(64, (3,3), activation='relu'))
        self.__nn.add(BatchNormalization())
        self.__nn.add(MaxPool2D((2, 2)))
        self.__nn.add(Dropout(0.5))

        #4. next we stretch the so far processed images into 1-d vectors
        self.__nn.add(Flatten())

        # MLP layers
        self.__nn.add(Dense(128, activation='relu'))
        self.__nn.add(BatchNormalization())
        self.__nn.add(Dense(64, activation='relu'))
        self.__nn.add(BatchNormalization())

        # Output layer
        self.__nn.add(Dense(1, activation='sigmoid'))

        self.__nn.compile(loss='binary_crossentropy', metrics=['accuracy'], **kwargs)


    def fit(self, *args, **kwargs):
        self.__history = self.__nn.fit(*args, **kwargs)


    def evaluate(self, test, n: int, batch_size: int):
        return self.__nn.evaluate(test, steps=n // batch_size, verbose=2)


    def save_weights_history(self):
        # milliseconds since Unix epoch
        ms = str(int(time.time()))
        self.__nn.save_weights("./checkpoints/" + "model_" + ms +
            ".weights.h5")

        with open("./history/" + "history_" + ms + ".json", "w") as file:
            json.dump(self.__history.history, file)

    def load_weights(self, checkpoint: str):
        self.__nn.load_weights("./checkpoints/" + checkpoint)
