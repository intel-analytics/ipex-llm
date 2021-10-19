#!/bin/bash

## Usage ################################
# ./ipynb2py <file-name without extension>
# Example:
# ipynb2py notebooks/neural_networks/rnn
#########################################
if [ $# -ne "1" ]; then
    echo "Usage: ./nb2script <file-name without extension>"
else
    cp $1.ipynb $1.tmp.ipynb
    sed -i 's/%%/#/' $1.tmp.ipynb
    sed -i 's/%pylab/#/' $1.tmp.ipynb

    jupyter nbconvert $1.tmp.ipynb --to python

    mv $1.tmp.py $1.py
    sed -i '1i# -*- coding: utf-8 -*-' $1.py
    sed -i '#!/usr/bin/python' $1.py
    rm $1.tmp.ipynb
fi