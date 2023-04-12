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
train_data = data.iloc[:400, 2:].astype(float).replace(0, 0.0001)
test_data = data.iloc[400:, 2:].astype(float).replace(0, 0.0001)

# Separar características de entrada y variable de destino
X_train = train_data.iloc[:, 1:].astype(float).replace(0, 0.0001)
y_train = train_data.iloc[:, 0].astype(float).replace(0, 0.0001)
X_test = test_data.iloc[:, 1:].astype(float).replace(0, 0.0001)
y_test = test_data.iloc[:, 0].astype(float).replace(0, 0.0001)

# Construir modelo de red neuronal

model = Sequential()
model.add(Dense(2, input_dim=9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajustar modelo
model.fit(X_train, y_train, epochs=50, batch_size=10)


# Evaluar precisión del modelo en datos de prueba
_, accuracy = model.evaluate(X_test, y_test)
print(y_test)
print('Precisión: %.2f' % (accuracy*100))
# Hace una predicción con los últimos datos de la prueba
last_X_test = X_test.iloc[-5, :]
last_X_test = np.array(last_X_test).reshape(1, -1)
prediction = model.predict(last_X_test)

# Imprime la predicción
print(f"La predicción es: {prediction[0][0]}")

