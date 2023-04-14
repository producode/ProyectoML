import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Cargar datos
data = pd.read_csv('stockData.csv', header=0, sep=',')

data = data.replace(',', '.', regex=True)


# Verifica si hay valores faltantes
if data.isnull().values.any():
    print("Hay valores faltantes en los datos.")
else:
    print("No hay valores faltantes en los datos.")

data = data.dropna()

# Verifica si hay valores faltantes
if data.isnull().values.any():
    print("Hay valores faltantes en los datos.")
else:
    print("No hay valores faltantes en los datos.")

# Elimina las filas que tengan algún cero en algún dato
data = data.loc[~(data==0).any(axis=1)]

print(data)

# Dividir los datos en conjuntos de entrenamiento y prueba
train_data = data.iloc[:400, 2:].astype(float)
test_data = data.iloc[400:, 2:].astype(float)

# Separar características de entrada y variable de destino
X_train = train_data.iloc[:, 1:].astype(float)
y_train = train_data.iloc[:, 0].astype(float)
X_test = test_data.iloc[:, 1:].astype(float)
y_test = test_data.iloc[:, 0].astype(float)

print(X_train)
print(X_test)

# Construir modelo de red neuronal

model = Sequential()
model.add(Dense(18, input_shape=(9,), activation='linear'))
model.add(Dense(9, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

# Compilar modelo
model.compile(loss='mean_squared_error', optimizer='Adadelta', metrics=['accuracy'])

# Ajustar modelo
model.fit(X_train, y_train, epochs=200, batch_size=10)


# Evaluar precisión del modelo en datos de prueba
_, accuracy = model.evaluate(X_test, y_test)
print(y_test)
print('Precisión: %.2f' % (accuracy*100))
# Hace una predicción con los últimos datos de la prueba
last_X_test = X_test.iloc[-1, :]
last_X_test = np.array(last_X_test).reshape(1, -1)
prediction = model.predict(last_X_test)

# Imprime la predicción
print(f"La predicción es: {prediction[0][0]}")

