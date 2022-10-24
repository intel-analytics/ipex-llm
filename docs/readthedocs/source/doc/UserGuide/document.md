# BigDL Documentation Guide: Writing Tips & Rules
## 1. Differentiate the syntax of reStructuredText and Markdown
Our documentation includes both reStructuredText (.rst) and Markdown (.md) files. They have different syntax, please make sure you do not mix the usage of them.

> Refer to [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) for reStructuredText syntax examples.

### 1.1 Location to use reStructuredText or Markdown in convention
| reStructuredText | Markdown |
| :----------------| :--------|
| - index pages (which includes or leads to pages with lower hierarchy in table of contents) <br> - **description comments in source code (for API documentation)**  | - other pages |

### 1.2 The places where reStructuredText and Markdown syntax are often confused

<table>
<tr><th></th><th>reStructuredText</th><th>Markdown</th></tr>
<tr><td>inline code</td>
<td>

```rst
``inline code``
```

</td>
<td>

```markdown
`inline code`
```

</td></tr>
<tr><td>hyperlinks</td>
<td>

```rst
`Relative Link text <relatve/path/to/the/file>`_ 
`Absolute Link text <https://www.example.com/>`_ 
```

</td>
<td>

```markdown
[Relative Link text](relatve/path/to/the/file)
[Absolute Link text](https://www.example.com/)
```

</td></tr>
<tr><td>italic</td>
<td>

```rst
`italic text`
*italic text*
```

</td>
<td>

```markdown
*italicized text*
```

</td></tr>
<tr><td>italic & bold</td>
<td>

not supported, needed help with css

</td>
<td>

```markdown
***italicized & bold text***
```

</td></tr>
</table>

### 1.3 Tips when adding comments in source code for API documentation
According to the [`sphinx.ext.autodoc` document](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc), "the docstrings must of course be written in correct reStructuredText". We need to make sure that we are using reStructuredText syntax in the source code comments for API documentation.

There are two special syntax often used in API documentation for parameter definition and return values. Let us take a snippet from Nano as an example:
```rst
:param use_ipex: (optional) if not None, then will only find the
       model with this specific ipex setting.
:param accuracy_criterion: (optional) a float represents tolerable
       accuracy drop percentage, defaults to None meaning no accuracy control.
:return: best model, corresponding acceleration option
```

Note that the following lines of one parameter/return definition should be indented to be rendered correctly.

> **Note**: To make sure the API documentation is rendered correctly, please always checked the rendered documentation when changed made to the docstrings.
## 2. Common reStructuredText components
<table>
<tr><td>Headers</td>
<td>

```rst
Header Level 1
=========================

Header Level 2
-------------------------

Header Level 3
~~~~~~~~~~~~~~~~~~~~~~~~~

Header Level 4
^^^^^^^^^^^^^^^^^^^^^^^^^
```

</td>
<td>

Note that the underline symbols should be at least as long as the header texts. <br>
<br>
Also, **we do not expect maually-added styles to headers.**<br>
<br>
You could refer to [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections) for more information on reStructuredText sections.

</td></tr>
<tr><td>Lists</td>
<td>

```rst
* A unordered list
* The second item of the unordered list
  with two lines

#. A numbered list

   1. A nested numbered list
   2. The second nested numbered list

#. The second item of 
   the numbered list
```

</td>
<td>

Note that the number of spaces indented depends on the markup. That is, if we use `* `/`#. `/`10. ` for the list, the indent after it should be indented by 2/3/4 spaces. <br>
<br>
Also note that blanks lines are needed around the nested list. <br>
<br>
You could refer to [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#lists-and-quote-like-blocks) for more information on reStructuredText lists.

</td></tr>
<tr><td>Note, <br> Warning, <br> Danger, <br> Tip, <br> Important, <br> See Also boxes</td>
<td>

```rst
.. note::

   This is a note box.

.. warning::

   This is a warning box.

.. danger::

   This is a danger box.

.. tip::

   This is a tip box.

.. important::

   This is an important box.

.. seealso::

   This is a see also box.
```

</td>
<td>

![image](https://user-images.githubusercontent.com/54161268/197151779-cd93772a-cfe5-414b-a9b7-9cf8e4453775.png)

</td></tr>
<tr><td>Code blocks</td>
<td>

```rst
.. code-block:: [language]

   some code in this language

.. code-block:: python

   some python code
```

</td>
<td>

All the supported language argument for syntax highlighting can be found [here](https://pygments.org/docs/lexers/).

</td>
</tr>
</table>

### 2.1 Use reStructuredText in Markdown files
You could embed reStructuredText into Markdown files through putting reStructuredText code into `eval_rst` code block:
~~~
```eval_rst
```
~~~
It could be useful if you want to use special boxes (e.g. note box) in your Markdown files.

## 3. Common Markdown components
<table>
<tr><td>Headers</td>
<td>

```markdown
# Header Level 1

## Header Level 2

### Header Level 3

#### Header Level 4
```

</td>
<td>

Note that **we do not expect maually-added styles to headers.**

</td></tr>
<tr><td>Lists</td>
<td>

```markdown
- A unordered list
- The second item of the unordered list
  with two lines

1. A numbered list
   * A nested unordered list
   * The second nested unordered list
2. The second item of 
   the numbered list
```

</td>
<td>

Note that the number of spaces indented depends on the markup. That is, if we use `- `/`1. `/`10. ` for the list, the indent after it should be indented by 2/3/4 spaces.

</td></tr>
<tr><td>Code blocks</td>
<td>

~~~markdow
```[language]
some code in this language
```

```python
some python code
```
~~~

</td>
<td>

All the supported language argument for syntax highlighting can be found [here](https://pygments.org/docs/lexers/).

</td>
</tr>
</table>

## 4. How to include Jupyter notebooks directly inside our documentation
If you want to include a Jupyter notebook into our documentation as an example, how-to guides, etc., you could just put it anywhere inside `BigDL/docs/readthedocs/source` dictionary, and link it into either `BigDL/docs/readthedocs/source/_toc.yml` or any index page.

However, if you want to render a Jupyter notebook located out of `BigDL/docs/readthedocs/source` dictionary into our documentation, the case is a little bit complicated. To do this, you need to add a file with `.nblink` extension into `BigDL/docs/readthedocs/source` dictionary, and link the `.nblink` file into `BigDL/docs/readthedocs/source/_toc.yml` or any index page.

The `.nblink` file should have the following structure:
```json
{
    "path": "relative/path/to/the/notebook/you/want/to/include"
}
```

> You could find [here](https://github.com/intel-analytics/BigDL/blob/main/docs/readthedocs/source/doc/Nano/Howto/Training/PyTorchLightning/accelerate_pytorch_lightning_training_ipex.nblink) for an example of `.nblink` usage inside our documentation.

### 4.4 How to hide a cell from rendering into our documentation

If you want to hide a notebook markdown/code cell from rendering into our documentation, you could simply add `"nbsphinx": "hidden"` into the cell's `metadata`.

> **Note**: Currently we could not individually hide the output/input code cell from rendering as they have the same `metadata`. 