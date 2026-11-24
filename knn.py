
# IMPORTAZIONE LIBRERIE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# CARICAMENTO DATASET

data = fetch_california_housing()

X = data.data
y = data.target

Xdf = pd.DataFrame(X, columns=data.feature_names)


# STANDARDIZZAZIONE

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# RICERCA DEL MIGLIOR K

k_range = range(1, 21)
mse_values = []

for k in k_range:

    knn_reg = KNeighborsRegressor(n_neighbors=k)
    knn_reg.fit(X_scaled, y)

    y_pred = knn_reg.predict(X_scaled)

    mse = mean_squared_error(y, y_pred)
    mse_values.append(mse)


# GRAFICO ELBOW METHOD

plt.figure(figsize=(8, 6))

plt.plot(
    k_range, mse_values,
    marker='o', linestyle='-',
    color='darkblue'
)

plt.title("Elbow Method - Scelta di K")
plt.xlabel("Numero di vicini (K)")
plt.ylabel("MSE")
plt.xticks(k_range)
plt.grid(linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("elbow_method_knn.png", dpi=300)
plt.show()


# ADDESTRAMENTO MODELLO DEFINITIVO

best_k = 5

knn_final = KNeighborsRegressor(n_neighbors=best_k)
knn_final.fit(X_scaled, y)

# Generiamo le previsioni
y_pred_final = knn_final.predict(X_scaled)

# Calcoliamo le metriche
mse_final = mean_squared_error(y, y_pred_final)
r2_final = r2_score(y, y_pred_final)
mae_final = mean_absolute_error(y, y_pred_final)

print("\nRISULTATI KNN FINALE")
print(f"K: {best_k}")
print(f"MSE: {mse_final:.4f}")
print(f"R2: {r2_final:.4f}")
print(f"MAE: {mae_final:.4f}")


# GRAFICO DI REGRESSIONE CON UNA FEATURE

# Selezioniamo AveRooms, terza feature (indice 2)
feature_index = 2

# Ordiniamo i dati in base alla feature scelta
sort_idx = X_scaled[:, feature_index].argsort()

X_plot = X_scaled[sort_idx, feature_index]
y_pred_plot = y_pred_final[sort_idx]

# Creiamo il grafico
plt.figure(figsize=(10, 6))

plt.scatter(
    X_scaled[:, feature_index],
    y,
    color='gray',
    alpha=0.5,
    label='Dati reali'
)

plt.plot(
    X_plot,
    y_pred_plot,
    color='red',
    linewidth=2,
    label='Regressione KNN'
)

plt.title(f'K={best_k} - KNN Regression: Target vs AveRooms')
plt.xlabel('AveRooms (standardizzato)')
plt.ylabel('Target')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("knn_regression.png", dpi=300)
plt.show()
