import os

import json
import argumentos
import pandas as pd

# Config
DIR_RESULTADOS_GRIDSEARCH = "resultados_gridsearch"
DIR_RESULTADOS_RAW = "resultados_raw"
DIR_RESULTADOS = "resultados"
NOME_ARQUIVO = f"resultados_{argumentos.CLASSIFICADOR}_{argumentos.DATASET}"
NOME_ARQUIVO_RESULTADOS_GRID_SEARCH = (
    f"resultados_{argumentos.CLASSIFICADOR}_{argumentos.DATASET}.csv"
)
NOME_ARQUIVO_MELHORES_HIPERPARAMETROS = (
    f"melhor_hp_{argumentos.CLASSIFICADOR}_{argumentos.DATASET}.json"
)

# Gerar caminhos
caminho_resultado = f"{DIR_RESULTADOS}/{NOME_ARQUIVO}.csv"
caminho_resultado_raw = f"{DIR_RESULTADOS_RAW}/{NOME_ARQUIVO}_{argumentos.ITERACAO}.csv"
caminho_resultado_gs = (
    f"{DIR_RESULTADOS_GRIDSEARCH}/{NOME_ARQUIVO_RESULTADOS_GRID_SEARCH}"
)
caminho_melhor_hp_gs = (
    f"{DIR_RESULTADOS_GRIDSEARCH}/{NOME_ARQUIVO_MELHORES_HIPERPARAMETROS}"
)

COLUNAS = {
    "acuracia": 1,
    "kappa": 3,
    "tempo_treino": 5,
    "tempo_teste": 6,
    "tempo_total": 7,
}


def salvar_resultados_normal():
    # Ler o resultado do teste e criar um registro
    resultado = []
    with open(caminho_resultado_raw) as arquivo_resultado_raw:
        resultado = arquivo_resultado_raw.read().splitlines()[-1].split(",")
    # Carregar csv
    dados_csv = {}
    if os.path.exists(caminho_resultado):
        dados_csv = pd.read_csv(caminho_resultado, index_col=0).to_dict()

    indice = int(argumentos.ITERACAO)

    for nome, indice_coluna in COLUNAS.items():
        if not nome in dados_csv:
            dados_csv[nome] = {}
        coluna = dados_csv[nome]
        coluna[indice] = float(resultado[indice_coluna])

    pd.DataFrame(data=dados_csv).to_csv(caminho_resultado, index_label="x")


def salvar_resultados_gridsearch(resultados, melhor_hp: dict):
    pd.DataFrame(resultados).to_csv(caminho_resultado_gs)
    with open(caminho_melhor_hp_gs, "w") as arquivo:
        arquivo.write(json.dumps(melhor_hp))


# Criar diretórios
if not os.path.isdir(DIR_RESULTADOS):
    os.mkdir(DIR_RESULTADOS)
if not os.path.isdir(DIR_RESULTADOS_RAW):
    os.mkdir(DIR_RESULTADOS_RAW)
if not os.path.isdir(DIR_RESULTADOS_GRIDSEARCH):
    os.mkdir(DIR_RESULTADOS_GRIDSEARCH)
