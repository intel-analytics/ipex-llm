# DPGANSimulator Example
This example shows how to generate synthetic data with similar distribution as training data with the fast and easy `DPGANSimulator` API provided by Chronos.

**Chronos will continuously improve the simulator functionalities in accuracy, performance and usability.**

## Introduction
`DPGANSimulator` adopt DoppelGANger raised in [Using GANs for Sharing Networked Time Series Data: Challenges, Initial Promise, and Open Questions](http://arxiv.org/abs/1909.13403). The method is data-driven unsupervised method based on deep learning model with GAN (Generative Adversarial Networks) structure. The model features a pair of seperate attribute generator and feature generator and their corresponding discriminators `DPGANSimulator` also supports a rich and comprehensive input data (training data) format and outperform other algorithms in many evalution metrics.

## Data preparation
We will use WWT (Wikipedia Web Traffic) dataset stated in the [paper](http://arxiv.org/abs/1909.13403) in this example. Please download the training data (`data_train.npz`) [here](https://drive.google.com/drive/folders/14x5f4Q34mlyZbDjADT8jbuOrgWyuUjFV).

`DPGANSimulator` also require the meta data for the training data, for WWT dataset, the meta data can be stated as following.
```python
# feature outputs
from bigdl.chronos.simulator.doppelganger.output import Output, OutputType, Normalization


feature_outputs = [Output(type_=OutputType.CONTINUOUS, dim=1, normalization=Normalization.MINUSONE_ONE)]

# attribute outputs
attribute_outputs = [Output(type_=OutputType.DISCRETE, dim=9),
                     Output(type_=OutputType.DISCRETE, dim=3),
                     Output(type_=OutputType.DISCRETE, dim=2)]
```

## Quick Start
You can easily run the example through
```bash
python dpgansimulator_wwt.py --datadir /path/to/data_train.npz
```
Typically, the example takes **40~60** mins to complete the training and evalution in this example.

Three `./*.png` files will be generated as reproduction of figure 1, 6, 19 in the [paper](http://arxiv.org/abs/1909.13403). A checkpoint will be saved in `./checkpoint` for the trained model.

## Options
- --cores: The number of cpu cores you want to use on each node. You can change it depending on your own cluster setting.
- --epoch: Max number of epochs to train in each trial.
- --datadir: Use local csv file by default.
- --plot_figures: Plot Figure 1, 6, 19 in the http://arxiv.org/abs/1909.13403
- --batch_size: Training batch size.
- --checkpoint_path: The checkpoint root dir.
- --checkpoint_every_n_epoch: Checkpoint per n epoch.
