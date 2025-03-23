#import Libraries 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#loading dataset
df = pd.read_csv('heart.csv')
# Showing 5 rows from the dataset
df.head()

#splitting dataset
X = df.drop('target', axis = 1)
y = df['target']
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Standardizing the data
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

#creating model 
model = Sequential([
    Dense(32, activation = 'relu', input_shape = (xtrain.shape[1],)),
    Dropout(0.1),	
    Dense(32, activation = 'relu'),
    Dropout(0.5),
    Dense(1, activation = 'sigmoid')
])

#compiling model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()


#training model 
model.fit(xtrain, ytrain, epochs = 20, batch_size = 16, validation_data = (xtest, ytest))

#Model Evaluation 
loss, accuracy = model.evaluate(xtest, ytest)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

#Save model
model.save('heartmodel.h5')

#loading Saved Model 
from tensorflow.keras.models import load_model
loaded_model = load_model('heartmodel.h5')

#Predicting with new data
#age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal
new_data = [[34, 0, 1, 118, 210, 0, 1, 192, 0, 0.7, 2, 0,2]]
# Preprocess the new data (scale it)
new_data_scaled = scaler.transform(new_data)
# Predict using the loaded model
prediction = loaded_model.predict(new_data_scaled)
# Convert the output from probability to class (0 or 1)
predicted_class = (prediction > 0.5).astype(int)
print(f'Prediction: {predicted_class[0][0]}')
