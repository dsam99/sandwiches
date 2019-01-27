from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D
from tensorflow import convert_to_tensor
from keras.models import load_model
import numpy as np

from img_convert import *


def create_model():
    '''
    Method to setup the Keras model
    '''

    # creating our keras model
    model = Sequential()

    # Adding layers
    model.add(Conv2D(64,
                     kernel_size=3,
                     padding="same",
                     activation='relu',
                     input_shape=(512, 384, 1)))

    model.add(Conv2D(32, kernel_size=3,
                     activation='relu'))

    # pooling layer
    model.add(Flatten())

    # final dense layer to produce output
    model.add(Dense(7, activation='softmax'))

    # Compiling model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_data():
    '''
    Method to setup training data
    '''

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    with open('data/labels1.csv') as f:
    	# creating training data
        data = f.readlines()
        num_files = len(data)

        for i in range(num_files - 2000):
            tuple_data = data[i].strip().split(",")
            image_file = "uploads/" + tuple_data[0]

	        # creating labels
            label = np.zeros(7)
            label[int(tuple_data[1])] = 1
            train_data.append(normalize_image(convert_image(image_file)))
            train_labels.append(label)
        
        for i in range(2000):
            tuple_data = data[i + (num_files - 2000)].strip().split(",")
            image_file = "uploads/" + tuple_data[0]
            # creating labels
            label = np.zeros(7)
            label[int(tuple_data[1])] = 1

            test_data.append(normalize_image(convert_image(image_file)))
            test_labels.append(label)

    train_data = np.array(train_data).reshape(-1, 512,384, 1)
    test_data = np.array(test_data).reshape(-1, 512,384, 1)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


def train_model(model):
    '''
    Method that loads the datasets and trains the model
    '''

    # loading in data size (512x384)
    train_data, train_labels, test_data, test_labels = create_data()
    print("Loaded data")
    # Train the model
    model.fit(train_data,
              train_labels,
              batch_size=10,
              validation_data=(test_data, test_labels),
              epochs=3)


def save_model(model):
    '''
    Method to save the current state of the keras model and delete the 
    existing model
    '''

    model.save('sandwich_model_dropout2.h5')
    del model

def predict_class(model, image_file):
    '''
    Method to predict the class of a new image
    '''

    new_image = normalize_image(convert_image(image_file))
    pred_in = np.array([new_image])

    pred_in = pred_in.reshape(1, 512, 384, 1)

    # make a prediction
    prediction = model.predict(pred_in)
    # show the inputs and predicted outputs
    print("X=%s, Predicted=%s" % (pred_in[0], prediction[0]))

    pred = prediction.argmax(axis=-1)
    return pred

def main():

    # creating keras model
    model = create_model()

    # training and saving model
    train_model(model)
    save_model(model)

    # loading model
    # print(predict_class(model, "/home/daniel-ritter/food_101/churros/125886.jpg"))

    


if __name__ == "__main__":
	main()
