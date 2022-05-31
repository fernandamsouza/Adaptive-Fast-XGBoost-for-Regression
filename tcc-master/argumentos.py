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
ITERACAO = args.iteracao
MAX_REGISTROS = args.maxregistros

