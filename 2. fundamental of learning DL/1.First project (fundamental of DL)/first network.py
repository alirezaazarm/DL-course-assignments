import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = 'some where in storage'
data = pd.read_excel(file_path)
x = data.iloc[:,:16]
target = data.iloc[:,16]

X_train0, X_test, y_train0, y_test = train_test_split(x,target)
X_train1, X_validation, y_train1, y_validation = train_test_split(X_train0, y_train0)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train1)
X_test_s = sc.transform(X_test)
X_validation_s = sc.transform(X_validation)


model = keras.models.Sequential([
    keras.layers.Dense(100, activation="sigmoid",name='hidden layer 1'),
    keras.layers.Dense(150, activation="sigmoid",name='hidden layer 2'),
    keras.layers.Dense(75, activation="sigmoid", name='hidden layer 3'),
    keras.layers.Dense(1,name='output layer')
])

model.compile(loss="mean_squared_error",
              optimizer="sgd",
              metrics=["mean_absolute_error"])

model.fit(X_train_s, y_train1, epochs=30,
          validation_data=(X_validation_s, y_validation))


print("Weights and biases of the layers after training the model: \n")
for layer in model.layers:
  print(layer.name)
  print("Weights")
  print("Shape: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
  print("Bias")
  print("Shape: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')