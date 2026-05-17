import joblib
import pandas as pd


modelo = joblib.load("models/bot.pkl")


colunas = [
    "cell_-2_-2", "cell_-2_-1", "cell_-2_0", "cell_-2_1", "cell_-2_2",
    "cell_-1_-2", "cell_-1_-1", "cell_-1_0", "cell_-1_1", "cell_-1_2",
    "cell_0_-2", "cell_0_-1", "cell_0_1", "cell_0_2",
    "cell_1_-2", "cell_1_-1", "cell_1_0", "cell_1_1", "cell_1_2",
    "cell_2_-2", "cell_2_-1", "cell_2_0", "cell_2_1", "cell_2_2",
    "global_density"
]


def pegar_valor(tabuleiro, linha, coluna):
    if linha < 0 or coluna < 0:
        return -2

    if linha >= len(tabuleiro) or coluna >= len(tabuleiro[0]):
        return -2

    return tabuleiro[linha][coluna]


def montar_entrada(tabuleiro, linha, coluna):
    valores = []

    for dl in range(-2, 3):
        for dc in range(-2, 3):
            if dl == 0 and dc == 0:
                continue

            valor = pegar_valor(tabuleiro, linha + dl, coluna + dc)
            valores.append(valor)

    total = len(tabuleiro) * len(tabuleiro[0])
    ocultas = 0

    for l in tabuleiro:
        for celula in l:
            if celula == -1:
                ocultas += 1

    densidade = ocultas / total
    valores.append(densidade)

    entrada = pd.DataFrame([valores], columns=colunas)

    return entrada


def prever_jogada_segura(tabuleiro_atual, modelo=modelo):
    melhor_jogada = None
    melhor_chance = -1

    for linha in range(len(tabuleiro_atual)):
        for coluna in range(len(tabuleiro_atual[linha])):

            if tabuleiro_atual[linha][coluna] == -1:
                entrada = montar_entrada(tabuleiro_atual, linha, coluna)
                chance = modelo.predict_proba(entrada)[0][1]

                if chance > melhor_chance:
                    melhor_chance = chance
                    melhor_jogada = (linha, coluna)

    if melhor_jogada is not None:
        return melhor_jogada

    return (0, 0)