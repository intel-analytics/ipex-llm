# Nano Installation

Note: For windows users, we recommend using Windows Subsystem for Linux 2 (WSL2) to run BigDL-Nano. Please refer to [Nano Windows install guide](../Howto/windows_guide.md) for instructions.


BigDL-Nano can be installed using pip and we recommend installing BigDL-Nano in a conda environment.

You can select bigdl-nano along with some dependencies specific to PyTorch or Tensorflow using the following panel.

```eval_rst
.. raw:: html

    <link rel="stylesheet" type="text/css" href="../../../_static/css/nano_installation_guide.css" />

    <div class="displayed">
      <table id="table-1">
        <tbody>
          <tr>
            <td colspan="1">FrameWork</td>
            <td colspan="2"><button id="pytorch">Pytorch</button></td>
            <td colspan="2"><button id="tensorflow">Tensorflow</button></td>
          </tr>
          <tr id="version" class="taller_tr">
            <td colspan="1">Version</td>
            <td colspan="1"><button id="pytorch_113">1.13</button></td>
            <td colspan="1"><button id="pytorch_112">1.12</button></td>
            <td colspan="1"><button id="pytorch_111">1.11</button></td>
            <td colspan="1"><button id="pytorch_110">1.10</button></td>
          </tr>
          <tr>
            <td colspan="1">Inference Optimization</td>
            <td colspan="2"><button id="inferenceyes">Yes</button></td>
            <td colspan="2"><button id="inferenceno">No</button></td>
            </td>
          </tr>
          <tr>
            <td colspan="1">Release</td>
            <td colspan="2"><button id="nightly">Nightly</button></td>
            <td colspan="2"><button id="stable">Stable</button></td>
          </tr>
          <tr class="tallet_tr">
            <td colspan="1">Install CMD</td>
            <td colspan="4" id="cmd">NA</td>
          </tr>
        </tbody>
      </table>
    </div>

    <script src="../../../_static/js/nano_installation_guide.js"></script>
```

We also partially support M-series chip users with no guarantee of acceleration with same API. Currently only tensorflow is experimentally supported.

```bash
conda create -n env python=3.8
conda activate env
conda install -c apple tensorflow-deps
pip install --pre --upgrade bigdl-nano[tensorflow]
```

```eval_rst
.. note::
    Since bigdl-nano is still in the process of rapid iteration, we highly recommend that you install nightly build version through the above command to facilitate your use of the latest features.

    For stable version, please refer to the document and installation guide `here <https://bigdl.readthedocs.io/en/v2.1.0/doc/Nano/Overview/nano.html>`_ .
```

```bash
conda create -n env
conda activate env
# select your preference in above panel to find the proper command to replace the below command, e.g.
pip install --pre --upgrade bigdl-chronos[pytorch]
# after installing bigdl-nano, you can run the following command to setup a few environment variables.
source bigdl-nano-init
```

The `bigdl-nano-init` scripts will export a few environment variable according to your hardware to maximize performance.

In a conda environment, when you run `source bigdl-nano-init` manually, this command will also be added to `$CONDA_PREFIX/etc/conda/activate.d/`, which will automaticly run when you activate your current environment.

In a pure pip environment, you need to run `source bigdl-nano-init` every time you open a new shell to get optimal performance and run `source bigdl-nano-unset-env` if you want to unset these environment variables.

---
