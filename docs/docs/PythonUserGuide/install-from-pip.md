## **NOTES**

- Pip install supports __Mac__ and __Linux__ platforms.
- Pip install only supports __local__ mode. Might support cluster mode in the future. For those who want to use BigDL in cluster mode, try to [install without pip](./install-without-pip.md).
- We've tested this package with __Python 2.7__ and __Python 3.5__. Only these two Python versions are supported for now.


## **Install BigDL-0.3.0.dev0**

Install BigDL release via pip (we tested this on pip 9.0.1)

**Remark:**

- You might need to add `sudo` if without permission for the installation.

- `pyspark` will be automatically installed first before installing BigDL if it hasn't been detected locally.
```bash
pip install --upgrade pip
pip install BigDL==0.3.0.dev0     # for Python 2.7
pip3 install BigDL==0.3.0.dev0    # for Python 3.5
```
