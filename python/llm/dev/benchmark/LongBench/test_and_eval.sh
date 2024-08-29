#! /bin/sh

export HF_ENDPOINT=https://hf-mirror.com

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
python ${SHELL_FOLDER}/pred_snap.py
python ${SHELL_FOLDER}/eval.pty