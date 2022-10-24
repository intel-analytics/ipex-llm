# BigDL Documentation
This is the repositary for BigDL documentation, which is hosted at https://bigdl.readthedocs.io/en/latest/
## Local build
### 1. Set up environment
To build BigDL documentation locally for testing purposes, it is recommended to create a conda environment with specified Python version:

```bash
conda create -n docs python=3.7
conda activate docs
```

Then inside [`BigDL/docs/readthedocs`](.) folder, install required packages:

```bash
pip install --upgrade -r requirements-rtd.txt
pip install -r requirements-doc.txt
```
> **Note**: `requirements-rtd.txt` is for reproducing ReadtheDocs deployment environment. No need to modify this file when adding new sphinx extension for our documents. New packages should be added in `requirements-doc.txt`.

### 2. Build the documentation
You can then build the documentation locally through:
```bash
make html
```
> **Tips**: If you meet building error `Notebook error: Pandoc wasn't found`, try `conda install pandoc` to resolve it.

> **Note**: The built files inside `docs/readthedocs/_build/html` dictionary should not be commited, they are only for testing purposes.

### 3. Test the documentation
To view the documentation locally, you could set up a testing server:
```bash
cd _build/html
python -m http.server 8000
```
The documentation can then be reached at [http://localhost:8000/](http://localhost:8000/).

> **Note**: If you are setting up the testing server on a remote machine, it is recommended to forward port `8000` through VSCode, so that you could reach [http://localhost:8000/](http://localhost:8000/) normally as on your local machine.