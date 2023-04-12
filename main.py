import tensorflow as tf
import numpy as np
import pandas
import matplotlib.pyplot as plt

csv_file = pandas.read_csv("stockData2.csv", index_col=False)
print(csv_file)
stockData = csv_file.iloc[:70, 3:]
result = csv_file.iloc[:70, 2]
ejemplo_prediccion = csv_file.iloc[71, 3:]
print(ejemplo_prediccion)

inputTest = [[2.0, 3.0], [8.0, 10.0], [1.0, 9.0], [7.0, 8.0], [5.0, 4.0]]
outputTest = np.array([6.0, 80.0, 9.0, 56.0, 20.0], dtype=float)

capas = tf.keras.layers.Dense(units=6, input_dim=2, activation='relu')
oculta1 = tf.keras.layers.Dense(units=5)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capas, oculta1, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss="mean_squared_error"
)

print("Entrenando...")
historial=modelo.fit(stockData, result, epochs=1000, verbose=False)
print("Finalizado")

plt.xlabel("# epoch")
plt.ylabel("Magnitud perdida")
plt.plot(historial.history["loss"])

prediccion = modelo.predict([3.0])
print(prediccion)
