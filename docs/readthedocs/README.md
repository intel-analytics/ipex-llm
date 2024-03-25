# IPEX-LLM Documentation
This is the repository for IPEX-LLM documentation, which is hosted at https://ipex-llm.readthedocs.io/en/latest/

## Local build
### 1. Set up environment
To build IPEX-LLM documentation locally for testing purposes, it is recommended to create a conda environment with specified Python version:

```bash
conda create -n docs python=3.7
conda activate docs
```

Then inside [`ipex-llm/docs/readthedocs`](.) folder, install required packages:

```bash
cd docs/readthedocs
# for reproducing ReadtheDocs deployment environment
pip install --upgrade pip "setuptools<58.3.0"
pip install --upgrade pillow mock==1.0.1 "alabaster>=0.7,<0.8,!=0.7.5" commonmark==0.9.1 recommonmark==0.5.0 sphinx sphinx-rtd-theme "readthedocs-sphinx-ext<2.3"

# for other documentation related dependencies
wget https://raw.githubusercontent.com/analytics-zoo/gha-cicd-env/main/python-requirements/requirements-doc.txt
pip install -r requirements-doc.txt
```
> **Note**: When adding new sphinx extensions for our documentation, the requirements file located [here](https://raw.githubusercontent.com/analytics-zoo/gha-cicd-env/main/python-requrirements/requirements-doc.txt) should be modified.
### 2. Build the documentation
You can then build the documentation locally through:
```bash
make html
```
> **Tips**: If you meet building error `Notebook error: Pandoc wasn't found`, try `conda install pandoc` to resolve it.

> **Note**: The built files inside `docs/readthedocs/_build/html` dictionary should not be committed, they are only for testing purposes.

### 3. Test the documentation
To view the documentation locally, you could set up a testing server:
```bash
cd _build/html
python -m http.server 8000
```
The documentation can then be reached at [http://localhost:8000/](http://localhost:8000/).

> **Note**: If you are setting up the testing server on a remote machine, it is recommended to forward port `8000` through VSCode, so that you could reach [http://localhost:8000/](http://localhost:8000/) normally as on your local machine.