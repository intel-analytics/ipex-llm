#!/bin/bash

if [ "$(uname -m)" = "x86_64" ]; then
    echo "

    Installed package of scikit-learn can be accelerated using scikit-learn-intelex.
    More details are available here: https://intel.github.io/scikit-learn-intelex

    For example:

        $ conda install scikit-learn-intelex
        $ python -m sklearnex my_application.py

    " >> "$PREFIX/.messages.txt"
fi
