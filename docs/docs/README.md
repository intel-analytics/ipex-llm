## **Readme**
Before we create web pages according to documents, environments need to be set.

**Follow the commands here:**
```
conda create --name py36 python=3.6
pip install mkdocs==0.16.3
cd analytics-zoo
docs/gen_site.py -p -s -m 8080
```
Then choose correct proxy and open the link of your ip address and port number. 

Tips:
* Please use Python 3.5 or 3.6.
* If AttributeError appears, check your pip version. The following command may be useful:

```
python -m pip install --user --upgrade pip==9.0.3
```
