# Friesian User Guide

## **Overview**
BigDL Friesian is an open source framework for building and deploying recommender systems and works with the other BigDL components including DLlib and Orca to provide end-to-end recommender solution on Intel CPU. 
It provides: 
1. An offline pipeline, including unified and easy-to-use APIs for [feature engineering](../QuickStart/offline.md), [examples](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example) of common recommender models.
2. A nearline pipeline of loading processed data into redis database.
3. An online framework of distributed serving.

## **Install**
Note: For windows Users, we recommend using Windows Subsystem for Linux 2 (WSL2) to run BigDL-Friesian. Please refer [here](./windows_guide.md) for instructions.

BigDL-Friesian can be installed using pip and we recommend installing BigDL-Friesian in a conda environment.

```bash
conda create -n bigdl python==3.7
conda activate bigdl
pip install bigdl-friesian
```

## **Quick Start**

### **Offline feature engineering**

### **Offline distributed training**

