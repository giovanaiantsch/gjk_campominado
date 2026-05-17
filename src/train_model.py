import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dados = pd.read_csv("dataset/minesweeper_dataset.csv")
X = dados.drop("safe", axis=1)
y = dados["safe"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

modelo = RandomForestClassifier(
    n_estimators=40,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

print("Treinando modelo...")
modelo.fit(X_treino, y_treino)
previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)
print(f"Acurácia: {acuracia * 100:.2f}%")
joblib.dump(modelo, "models/bot.pkl")
print("Modelo salvo em models/bot.pkl")