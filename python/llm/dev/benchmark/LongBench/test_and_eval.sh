#! /bin/sh

export HF_ENDPOINT=https://hf-mirror.com

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
python ${SHELL_FOLDER}/pred.py
python ${SHELL_FOLDER}/eval.py