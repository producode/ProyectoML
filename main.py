import tensorflow as tf
import numpy as np
import pandas
import matplotlib.pyplot as plt

csv_file = pandas.read_csv("stockData.csv", index_col=False)
print(csv_file)
stockData = csv_file.iloc[:70, 1:].to_numpy(dtype=float)
result = csv_file.iloc[:70, 0].to_numpy(dtype=float)
ejemplo_prediccion = csv_file.iloc[71, 1:].to_numpy(dtype=float)
print(ejemplo_prediccion)

capas = tf.keras.layers.Dense(units=42, input_shape=[42])
modelo = tf.keras.Sequential([capas])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss="mean_squared_error"
)

print("Entrenando...")
historial=modelo.fit(stockData, result, epochs=100, verbose=False)
print("Finalizado")

plt.xlabel("# epoch")
plt.ylabel("Magnitud perdida")
plt.plot(historial.history["loss"])

prediccion = modelo.predict(ejemplo_prediccion)
print(prediccion)
