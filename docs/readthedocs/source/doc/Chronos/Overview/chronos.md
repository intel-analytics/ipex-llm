# Chronos User Guide

### **1. Overview**
_BigDL-Chronos_ (_Chronos_ for short) is an application framework for building a fast, accurate and scalable time series analysis application.

You can use _Chronos_ to:

```eval_rst
.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :class-footer: sd-bg-light

        **Forecasting**
        ^^^

        .. image:: ../Image/forecasting.svg
            :width: 200
            :alt: Alternative text

        +++

        Predict future using history data.

    .. grid-item-card::
        :class-footer: sd-bg-light

        **Anomaly Detection**
        ^^^

        .. image:: ../Image/anomaly_detection.svg
            :width: 200
            :alt: Alternative text

        +++

        Discover unexpected items in data.
    
    .. grid-item-card::
        :class-footer: sd-bg-light

        **Simulation**
        ^^^

        .. image:: ../Image/simulation.svg
            :width: 200
            :alt: Alternative text

        +++

        Generate similar data as history data.
```

---
### **2. Install**

```eval_rst
.. raw:: html

    <link rel="stylesheet" type="text/css" href="../../../_static/css/chronos_installation_guide.css" />

    <div class="displayed">
    
        <table id="table-1">
            <tbody>
                <tr>
                    <td colspan="1">Functionality</td>
                    <td colspan="1"><button id="Forecasting">Forecasting</button></td>
                    <td colspan="2"><button id="Anomaly" style="font-size: 15px">Anomaly Detection</button></td>
                    <td colspan="1"><button id="Simulation">Simulation</button></td>
                </tr>
                <tr id="model">
                    <td colspan="1">Model</td>
                    <td colspan="1"><button id="Deep_learning_models" style="font-size: 13px;">Deep learning models</button></td>
                    <td colspan="2"><button id="Prophet">Prophet</button></td>
                    <td colspan="1"><button id="ARIMA">ARIMA</button></td>
                </tr>
                <tr>
                    <td colspan="1">DL<br>framework</td>
                    <td colspan="2"><button id="pytorch"
                            title="Use PyTorch as deep learning models' backend. Most of the model support and works better under PyTorch.">PyTorch (Recommended)</button>
                    </td>
                    <td colspan="2"><button id="tensorflow"
                            title="Use Tensorflow as deep learning models' backend.">Tensorflow</button></td>
                </tr>
                <tr>
                    <td colspan="1">OS</td>
                    <td colspan="2"><button id="linux" title="Ubuntu/CentOS is recommended">Linux</button></td>
                    <td colspan="2"><button id="win" title="WSL is needed for Windows users">Windows</button></td>
                </tr>

                <tr>
                    <td colspan="1">Auto Tuning</td>
                    <td colspan="2" title="I don't need any hyperparameter auto tuning feature."><button
                            id="automlno">No</button></td>
                    <td colspan="2" title="I need chronos to help me tune the hyperparameters."><button
                            id="automlyes">Yes</button></td>
                </tr>


                <tr>
                    <td colspan="1">Hardware</td>
                    <td colspan="2"><button id="singlenode" title="For users use laptop/single node server.">Single
                            node</button></td>
                    <td colspan="2"><button id="cluster" title="For users use K8S/Yarn Cluster.">Cluster</button></td>
                </tr>

                <tr>
                    <td colspan="1">Package</td>
                    <td colspan="2"><button id="pypi" title="For users use pip to install chronos.">Pip</button></td>
                    <td colspan="2"><button id="docker" title="For users use docker image.">Docker</button></td>
                </tr>

                <tr>
                    <td colspan="1">Version</td>
                    <td colspan="2"><button id="nightly"
                            title="For users would like to try chronos's latest feature">Nightly (2.1.0b)</button>
                    </td>
                    <td colspan="2"><button id="stable"
                            title="For users would like to deploy chronos in their production">Stable (2.0.0)</button></td>
                </tr>

                <tr>
                    <td colspan="1">Install CMD</td>
                    <td colspan="4">
                        <div id="cmd" style="text-align: left;">NA</div>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>

    <script src="../../../_static/js/chronos_installation_guide.js"></script> 
```

</br>

#### **2.1 Pypi**
When you install `bigdl-chronos` from PyPI. We recommend to install with a conda virtual environment. To install Conda, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).
```bash
conda create -n my_env python=3.7 setuptools=58.0.4
conda activate my_env
# click the installation panel above to find which installation option to use
pip install --pre --upgrade bigdl-chronos[pytorch]  # or other options you may want to use
source bigdl-nano-init  # accelerate the conda env
```

#### **2.2 Tensorflow backend**
Tensorflow is one of the supported backend of Chronos in nightly release version, while it can not work alone without pytorch in Chronos for now. We will fix it soon. If you want to use tensorflow backend, please
```bash
pip install --pre --upgrade bigdl-nano[tensorflow]
```
**after you install the pytorch backend chronos.**

#### **2.3 OS and Python version requirement**

```eval_rst
.. note:: 
    **Supported OS**:

     Chronos is thoroughly tested on Ubuntu (16.04/18.04/20.04), and should works fine on CentOS. If you are a Windows user, the most convenient way to use Chronos on a windows laptop might be using WSL2, you may refer to https://docs.microsoft.com/en-us/windows/wsl/setup/environment or just install a ubuntu virtual machine.
```
```eval_rst
.. note:: 
    **Supported Python Version**:

     Chronos only supports Python 3.7.2 ~ latest 3.7.x. We are validating more Python versions.
```

---


### **3. Which document to see?**

```eval_rst
.. grid:: 2
    :gutter: 1

    .. grid-item-card::
        :class-footer: sd-bg-light

        **Quick Tour**
        ^^^

        You may understand the basic usage of Chronos' components and learn to write the first runnable application in this quick tour page.

        +++
        `Quick Tour <./quick-tour.html>`_

    .. grid-item-card::
        :class-footer: sd-bg-light

        **User Guides**
        ^^^

        Our user guides provide you with in-depth information, concepts and knowledges about Chronos.

        +++

        `Data <./data_processing_feature_engineering.html>`_ / 
        `Forecast <./forecasting.html>`_ / 
        `Detect <./anomaly_detection.html>`_ / 
        `Simulate <./simulation.html>`_

.. grid:: 2
    :gutter: 1

    .. grid-item-card::
        :class-footer: sd-bg-light

        **How-to-Guide** / **Example**
        ^^^

        If you are meeting with some specific problems during the usage, how-to guides are good place to be checked.
        Examples provides short, high quality use case that users can emulated in their own works.

        +++

        `How-to-Guide <../Howto/index.html>`_ / `Example <../QuickStart/index.html>`_

    .. grid-item-card::
        :class-footer: sd-bg-light

        **API Document**
        ^^^

        API Document provides you with a detailed description of the Chronos APIs. 

        +++

        `API Document <../../PythonAPI/Chronos/index.html>`_

```
