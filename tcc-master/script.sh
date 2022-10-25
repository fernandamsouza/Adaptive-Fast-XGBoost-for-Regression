#!/bin/bash

COMANDO_PYTHON="python3.9"
ARQUIVO_TESTES="adaptive_xgboost_execution.py"
ARQUIVO_GRIDSEARCH="grid_cv.py"
MAX_REGISTROS=500000
QNT_X=5
DATASETS=("abrupto" "gradual" "incremental")
CLASSIFICADORES=("AXGBRegRD" "AXGBRegR" "AXGBRegSD" "AXGBRegS" "HTR" "KNN" "HTRA" "ARFReg")
CLASSIFICADORES_ALL=("ALL")
DIR_RESULTADOS_GS="resultados_script_final"

mkdir -p logs

for classificador in ${CLASSIFICADORES_ALL[@]}
do
    for dataset in ${DATASETS[@]}
    do
        for x in $(seq $QNT_X)
        do
            echo "Experimento usando $dataset com $classificador ($x/$QNT_X)"
            $COMANDO_PYTHON $ARQUIVO_TESTES --classificador=$classificador --dataset=$dataset --iteracao=$x --maxregistros=$MAX_REGISTROS
        done
    done
done
