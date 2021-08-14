from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        self.model = models.Sequential()
        self.model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    def train(self, epochs, batch_size, data, validation):
        history = self.model.fit(data[0],
                            data[1],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=validation)
        return history

    def get_model(self):
        return self.model

class Data:
    def __init__(self, size):
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=size)
        self.x_train = self.vectorize_sequence(train_data)
        self.x_test = self.vectorize_sequence(test_data)

        self.y_train = np.asarray(train_labels).astype("float32")
        self.y_test = np.asarray(test_labels).astype("float32")

    def decode_review(self,number):
        if number > 9999:
            return None
        word_index = imdb.get_word_index()
        reverse_word_index = dict(
            [(value, key) for (key, value) in word_index.items()])
        decoded_review = ' '.join(
            [reverse_word_index.get(i - 3, '?') for i in train_data[number]])
        return decoded_review

    def vectorize_sequence(self, sequence, dimension=10000):
        results = np.zeros((len(sequence),dimension))
        for i, sequence in enumerate(sequence):
            results[i, sequence] = 1
        return results

    def get_datapack(self, size):
        x_val = self.x_train[:size]
        partial_x_train = self.x_train[size:]

        y_val = self.y_train[:size]
        partial_y_train = self.y_train[10000:]

        return (partial_x_train, partial_y_train), (x_val, y_val)

    def get_prediction_set(self):
        return self.x_test


def visualize_loss(history, epochs_count):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, epochs_count+1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
