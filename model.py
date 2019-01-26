from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D
from tensorflow import convert_to_tensor
import img_convert


# Getting input data

training_examples = img_convert.convert_directory("/home/daniel-ritter/food_101/churros")

# Turns all training examples from numpy arrays into tensors
for example in training_examples:
    example = convert_to_tensor(example)

print(training_examples[0])
model = Sequential()

# Adding layers 

model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape = (None,None,1)))
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(7,activation='softmax'))


# Compiling model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

training_labels = None
test_examples = None
test_labels = None

# Train the model
model.fit(training_examples,training_labels,validation_data=(test_examples,test_labels),epochs=1)

