import argparse
from ast import parse
import json
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "--classificador",
    "-c",
    help="Nome do classificador",
    type=str,
    default="axgb",
)
parser.add_argument(
    "--dataset",
    "-d",
    help="Nome do dataset",
    type=str,
    default="sea_a",
)
parser.add_argument(
    "--hiperparametros",
    "-p",
    help="Caminho do arquivos de hiper-parâmetros do classificador em JSON",
    type=str,
    default="",
)
parser.add_argument(
    "--iteracao",
    "-i",
    help="Número da iteração",
    type=int,
    default=1,
)
parser.add_argument(
    "--maxregistros",
    "-m",
    help="Máximo do registros",
    type=int,
    default=1_000_000,
)

args = parser.parse_args()

CLASSIFICADOR = args.classificador
DATASET = args.dataset
HIPER_PARAMETROS = {}
ITERACAO = args.iteracao
MAX_REGISTROS = args.maxregistros

# Carregar arquivo de hiperparametros
if args.hiperparametros != "":
    try:
        with open(args.hiperparametros) as arquivo:
            HIPER_PARAMETROS = json.loads(arquivo.read())
    except IOError:
        print(f"{sys.argv[0]} erro: arquivo {args.hiperparametros} não existe")
    except json.JSONDecodeError:
        print(
            f"{sys.argv[0]} erro: não foi possível ler {args.hiperparametros} pois o arquivo não está formatado como JSON corretamente"
        )
