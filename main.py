from neural import *

if __name__ == "__main__":
    epochs = 15
    data_instance = Data(10000)
    data, validation = data_instance.get_datapack(10000)
    network = NeuralNetwork()
    history = network.train(epochs=epochs, batch_size=512, data=data, validation=validation)

    model = network.get_model()
    print(model.predict(data_instance.x_test))
    model.save('keras_model')