# BigDL Documentation

All official released docs are available in [BigDL Docs](https://bigdl-project.github.io/)


Our documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation:

1. install MkDocs: `pip install mkdocs`
2. `cp` the folder 'readthedocs/' from https://github.com/helenlly/bigdl-project.github.io to "docs/"
3. `cd` to the 'docs/'folder and  make sure 2 folders and 1 file existing: 
    - 'docs/'    
    - 'readthedocs/'    
    - 'mkdocs.yml'
4. run:   
    - `mkdocs serve`    # Starts a local webserver and you can view in:  [localhost:8000](localhost:8000), e.g. http://127.0.0.0:8000    
    - `mkdocs build`    # Builds a static site in "site" directory, you can view site/index.html directly
