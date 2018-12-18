## **Readme**
Before we create web pages according to documents, environments need to be set.

**Follow the commands here:**
```
conda create --name py27 python=2.7 
pip install mkdocs==0.16.3
cd analytics-zoo
docs/gen_site.py -p -s -m 8080
```
Then choose correct proxy and open the link of your ip address and port number. 

Tips:
* Please use python27, for now python3 is not supported.
* If AttributeError appears, check your pip version. The following command may be useful:

```
python -m pip install --user --upgrade pip==9.0.3
```

