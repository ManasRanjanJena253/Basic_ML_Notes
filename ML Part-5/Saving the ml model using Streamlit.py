# This code has to be written after the coding and training of our ml model.
#Saving the trained model


import pickle
import numpy as np
from sklearn.svm import SVC
import streamlit

classifier = SVC()
filename = 'trained_model.sav'   # The name with which our ml model will be saved using streamlit.
pickle.dump(classifier, open(filename, 'wb'))   # Classifier is the name of the model or the variable in which the ml model was stored.
# The second parameter suggests we are opening the file with the motive of writing in binary as suggested by 'wb'.
# This function will save our code into a file named as the name provided to the variable filename.
# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))  # 'rb' Signifies reading the binary file.


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')

