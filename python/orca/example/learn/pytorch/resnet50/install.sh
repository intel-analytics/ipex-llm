#!/usr/bin/env bash

conda install pytorch torchvision cpuonly -c pytorch
pip install intel_extension_for_pytorch
pip install --pre --upgrade bigdl-orca-spark3[ray]