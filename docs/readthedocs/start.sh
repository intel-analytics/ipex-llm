#!/bin/bash
conda activate docs
cd /home/arda/guancheng/BigDL/docs/readthedocs
make html
cd _build/html
python -m http.server 8000
