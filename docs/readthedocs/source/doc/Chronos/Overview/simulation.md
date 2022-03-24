# Generate Synthetic Sequential Data Overview

Chronos provides simulators to generate synthetic time series data for users who want to conquer limited data access in a deep learning/machine learning project or only want to generate some synthetic data to play with.

```eval_rst
.. note::
    DPGANSimulator is the only simulator chronos provides at the moment, more simulators are on their way.
```

## **1. DPGANSimulator**
`DPGANSimulator` adopt DoppelGANger raised in [Using GANs for Sharing Networked Time Series Data: Challenges, Initial Promise, and Open Questions](http://arxiv.org/abs/1909.13403). The method is data-driven unsupervised method based on deep learning model with GAN (Generative Adversarial Networks) structure. The model features a pair of seperate attribute generator and feature generator and their corresponding discriminators `DPGANSimulator` also supports a rich and comprehensive input data (training data) format and outperform other algorithms in many evalution metrics.

Users may refer to detailed [API doc](../../PythonAPI/Chronos/simulator.html#module-bigdl.chronos.simulator.doppelganger_simulator).
