import numpy as np
import joblib

modelo = joblib.load("models/bot.pkl")
def preparar_tabuleiro(tabuleiro):
    dados = np.array(tabuleiro)
    dados = dados.reshape(1, -1)
    return dados

def prever_jogada_segura(tabuleiro_atual, modelo=modelo):
    entrada = preparar_tabuleiro(tabuleiro_atual)
    probabilidades = modelo.predict_proba(entrada)[0]
    melhores = np.argsort(probabilidades)[::-1]

    for posicao in melhores:
        linha = posicao // 8
        coluna = posicao % 8
        if tabuleiro_atual[linha][coluna] == -1:
            return (linha, coluna)

    for linha in range(8):
        for coluna in range(8):
            if tabuleiro_atual[linha][coluna] == -1:
                return (linha, coluna)

    return (0, 0)