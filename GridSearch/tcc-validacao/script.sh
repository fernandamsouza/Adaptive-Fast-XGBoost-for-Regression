#!/bin/bash

COMANDO_PYTHON="python3"
ARQUIVO_TESTES="adaptive_xgboost_example.py"
ARQUIVO_GRIDSEARCH="grid_cv.py"
MAX_REGISTROS=500000
QNT_X=5
DATASETS=("pol")
DATASET_GS="pol"
CLASSIFICADORES=("axgb_reset" "axgb_sem_reset")
DIR_RESULTADOS_GS="resultados_gridsearch"

mkdir -p logs

for classificador in ${CLASSIFICADORES[@]}
do
    echo "Executando GridSearch no classificador $classificador"
    $COMANDO_PYTHON $ARQUIVO_GRIDSEARCH --classificador=$classificador --dataset=$DATASET_GS --maxregistros=$MAX_REGISTROS
    echo "Realizando experimentos"
    for dataset in ${DATASETS[@]}
    do
        for x in $(seq $QNT_X)
        do
            echo "Experimento usando $dataset ($x/$QNT_X)"
            $COMANDO_PYTHON $ARQUIVO_TESTES --classificador=$classificador --dataset=$dataset --hiperparametros="$DIR_RESULTADOS_GS/melhor_hp_${classificador}_$DATASET_GS.json" --iteracao=$x --maxregistros=$MAX_REGISTROS
        done
    done
done
