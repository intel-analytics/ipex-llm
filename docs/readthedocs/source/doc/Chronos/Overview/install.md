# Chronos User Guide

---
### **Install**

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
                            title="For users would like to try chronos's latest feature">Nightly (2.2.0b)</button>
                    </td>
                    <td colspan="2"><button id="stable"
                            title="For users would like to deploy chronos in their production">Stable (2.1.0)</button></td>
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

#### **2.2 OS and Python version requirement**

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

