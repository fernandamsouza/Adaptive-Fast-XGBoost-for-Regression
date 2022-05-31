#!/bin/bash

COMANDO_PYTHON="python3.9"
ARQUIVO_TESTES="adaptive_xgboost_example.py"
ARQUIVO_GRIDSEARCH="grid_cv.py"
MAX_REGISTROS=500000
QNT_X=1
DATASETS=("abrupto" "gradual" "incremental" "real")
CLASSIFICADORES=("AXGBRegRD" "AXGBRegR" "AXGBRegSD" "AXGBRegS" "HTR" "KNN" "HTRA" "ARFReg")
DIR_RESULTADOS_GS="resultados_script_final"

mkdir -p logs

for classificador in ${CLASSIFICADORES[@]}
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
