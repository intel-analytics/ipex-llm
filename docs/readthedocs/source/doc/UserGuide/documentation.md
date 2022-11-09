# Documentation Guide

Here list several writing tips and guidelines you could refer to if you want to add/modify documents for BigDL documentation. The source code of our documentation is available [here](https://github.com/intel-analytics/BigDL/tree/main/docs/readthedocs).

```eval_rst
.. tip::

   You could refer `here <https://github.com/intel-analytics/BigDL/blob/main/docs/readthedocs/README.md>`_ if you would like to test your local changes to BigDL documentation.
```

## 1. How to add a new document
### 1.1 Decide whether to add a reStructuredText (`.rst`) file or a CommonMark (`.md`) file
In our documentation, both reStructuredText (`.rst`) and CommonMark (`.md`) files are allowed to use. In convension, we use `.rst` file in index pages, and `.md` files for other pages.

Here shows an overview of our documentation structure tree:

```eval_rst
.. graphviz::

   digraph DocStructure {
      graph [tooltip=" " splines=ortho]
      node [color="#0171c3" shape=box fontname="Arial" fontsize=12 tooltip=" "]
      edge [tooltip=" "]
      
      N1 [label="BigDL (.rst)" style=filled fontcolor="#ffffff"]
      
      N1_1 [label="User guide (.rst)" style=filled fontcolor="#ffffff"]
      N1_2 [label="Powered by (.md)" style=rounded]
      N1_3 [label="Orca (.rst)" style=filled fontcolor="#ffffff"]
      N1_4 [label="Nano (.rst)" style=filled fontcolor="#ffffff"]
      N1_5 [label="DLlib (.rst)" style=filled fontcolor="#ffffff"]
      N1_6 [label="Chronos (.rst)" style=filled fontcolor="#ffffff"]
      N1_7 [label="Fresian (.rst)" style=filled fontcolor="#ffffff"]
      N1_8 [label="PPML (.rst)" style=filled fontcolor="#ffffff"]
      N1_9 [label="..." shape=plaintext]
      
      N1_1_1 [label="Python (.md)" style=rounded]
      N1_1_2 [label="Scala (.md)" style=rounded]
      N1_1_3 [label="..." shape=plaintext]
      
      N1_8_1 [label="PPML Intro. (.md)" style=rounded]
      N1_8_2 [label="User Guide (.md)" style=rounded]
      N1_8_3 [label="Tutorials (.rst)" style="filled" fontcolor="#ffffff"]
      N1_8_4 [label="..." shape=plaintext]
      
      
      N1_8_3_1 [label="..." shape=plaintext]
      
      N1_3_1 [label="..." shape=plaintext]
      N1_4_1 [label="..." shape=plaintext]
      N1_5_1 [label="..." shape=plaintext]
      N1_6_1 [label="..." shape=plaintext]
      N1_7_1 [label="..." shape=plaintext]
      
      N1 -> N1_1
      N1 -> N1_2
      N1 -> N1_3 -> N1_3_1
      N1 -> N1_4 -> N1_4_1
      N1 -> N1_5 -> N1_5_1
      N1 -> N1_6 -> N1_6_1
      N1 -> N1_7 -> N1_7_1
      N1 -> N1_8
      N1 -> N1_9
      
      N1_1 -> N1_1_1
      N1_1 -> N1_1_2
      N1_1 -> N1_1_3
      
      N1_8 -> N1_8_1
      N1_8 -> N1_8_2
      N1_8 -> N1_8_3 -> N1_8_3_1
      N1_8 -> N1_8_4
   }
```

Index pages (nodes filled with blue) are the ones supposed to lead to further pages. In the structure above, they are nodes with descendants.

```eval_rst
.. note::
   
   In convension, we use ``.rst`` file for index pages becuase various web components (such as cards, note boxes, tabs, etc.) are more straightforward to be inserted in our documentation through reStructuredText. And it is a common case in our documentation that index pages include various web components.
```

### 1.2 Add the new document to the table of contents (ToC)
For clear navigation purposes, it is recommended to put the document in the ToC. To do this, you need to insert the relative path to the newly-added file into the [`_toc.yml`](https://github.com/intel-analytics/BigDL/blob/main/docs/readthedocs/source/_toc.yml) file, according to its position in the structure tree.

```eval_rst
.. tip::

   When adding a new document, you should always check whether to put relative link directing to it inside its parent index page, or inside any other related pages.

.. warning::

   According to `sphinx-external-toc <https://sphinx-external-toc.readthedocs.io/en/latest/user_guide/sphinx.html#basic-structure>`_ document, "each document file can only occur once in the ToC".
```

For API related documents, we still use in-file `.. toctree::` directives instead of putting them inside `_toc.yml`. You could refer [here](https://github.com/intel-analytics/BigDL/tree/main/docs/readthedocs/source/doc/PythonAPI) for example usages.

## 2. Differentiate the syntax of reStructuredText and CommonMark
As mentioned above, our documentation includes both `.rst` and `.md` files. They have different syntax, please make sure you do not mix the usage of them.

```eval_rst
.. seealso::

   You could refer `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ for reStructuredText syntax examples, and `here <https://spec.commonmark.org/>`_ for CommonMark specifications.
```

Here list several use cases where syntax in `.rst` and `.md` are often confused:

<table class="table bigdl-documentation-guide-tables">
<tr><th></th><th>reStructuredText</th><th>CommonMark</th></tr>
<tr><td>Inline code</td>
<td>

```rst
``inline code``
```
</td>
<td>

```md
`inline code`
```

</td></tr>
<tr><td>Hyperlinks</td>
<td>

```rst
`Relative link text <relatve/path/to/the/file>`_ 
`Absolute link text <https://www.example.com/>`_ 
```

</td>
<td>

```md
[Relative link text](relatve/path/to/the/file)
[Absolute link text](https://www.example.com/)
```

</td></tr>
<tr><td>Italic</td>
<td>

```rst
`italicized text`
*italicized text*
```
</td>
<td>

```md
*italicized text*

```

</td></tr>
<tr><td>Italic & bold</td>
<td>

Not supported, needed help with css

</td>
<td>

```md
***italicized & bold text***
```

</td></tr>
</table>

```eval_rst
.. note::

   When linking to a ``.rst`` file in a ``.md`` file, replace the ``.rst`` with ``.html`` in the relative path to avoid errors.
   That is, if you want to link to the ``example.rst`` in a ``.md`` file, use 

   .. code-block:: md

      [Example](relatve/path/to/example.html)
```

### 2.1 Tips when adding docstrings in source code for API documentation
According to the [`sphinx.ext.autodoc`](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc) document, docstrings should be written in reStructuredText. We need to make sure that we are using reStructuredText syntax in the source code docstrings for API documentation.

There are two [field lists](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#field-lists) syntax often used in API documentation for parameter definition and return values. Let us take a snippet from [`bigdl.nano.pytorch.InferenceOptimizer.get_best_model`](../PythonAPI/Nano/pytorch.html#bigdl.nano.pytorch.InferenceOptimizer.get_best_model) as an example:
```rst
:param use_ipex: (optional) if not None, then will only find the
       model with this specific ipex setting.
:param accuracy_criterion: (optional) a float represents tolerable
       accuracy drop percentage, defaults to None meaning no accuracy control.
:return: best model, corresponding acceleration option
```

```eval_rst
.. important::

   The following lines of one parameter/return definition should be indented to be rendered correctly.

.. tip::

   Please always check whether corresponding API documentation is correctly rendered when changes made to the docstrings.
```
## 3. Common components in `.rst` files

<table class="table bigdl-documentation-guide-tables">
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

Note that the underline symbols should be at least as long as the header texts.

Also, **we do not expect maually-added styles to headers.**

You could refer [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections) for more information on reStructuredText sections.

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

Note that the number of spaces indented depends on the markup. That is, if we use '* '/'#. '/'10. ' for the list, the following contents belong to the list or the nested lists after it should be indented by 2/3/4 spaces.

Also note that blanks lines are needed around the nested list.

You could refer [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#lists-and-quote-like-blocks) for more information on reStructuredText lists.

</td></tr>
<tr><td>

Note, <br>
Warning, <br>
Danger, <br>
Tip, <br>
Important, <br>
See Also <br>
boxes

</td>
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

```eval_rst
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

</td></tr>
<tr><td>Tabs</td>
<td>

```rst
.. tabs::

   .. tab:: Title 1

      Contents for tab 1

   .. tab:: Title 2

      Contents for tab 2

      .. code-block:: python

         some python code
```

</td>
<td>

```eval_rst
.. tabs::

   .. tab:: Title 1

      Contents for tab 1

   .. tab:: Title 2

      Contents for tab 2

      .. code-block:: python

         some python code
```

You could refer [here](https://sphinx-tabs.readthedocs.io/en/v3.4.0/) for more information on the usage of tabs.

</td></tr>
<tr><td>Cards in grids</td>
<td>

```rst
.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card::

      **Header**
      ^^^
      A normal card.
      +++
      :bdg-link:`Footer <relatve/path>`

   .. grid-item-card::
      :link: https://www.example.com/
      :class-card: bigdl-link-card

      **Header**
      ^^^
      A link card.
      +++
      Footer
```

</td>
<td>

```eval_rst
.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card::

      **Header**
      ^^^
      A normal card.
      +++
      :bdg-link:`Footer <relatve/path>`

   .. grid-item-card::
      :link: https://www.example.com/
      :class-card: bigdl-link-card

      **Header**
      ^^^
      A link card.
      +++
      Footer
```

You could refer [here](https://sphinx-design.readthedocs.io/en/furo-theme/cards.html) for more information on the usage of cards, and [here](https://sphinx-design.readthedocs.io/en/furo-theme/grids.html#placing-a-card-in-a-grid) for cards in grids.

Note that `1 2 2 2` defines the number of cards per row in different screen sizes (from extra-small to large).

</td></tr>
<tr><td>

[Mermaid](https://mermaid-js.github.io/) digrams

</td><td>

```rst
.. mermaid::
   
   flowchart LR
      A(Node A)
      B([Node B])

      A -- points to --> B
      A --> C{{Node C}}

      classDef blue color:#0171c3;
      class B,C blue;
```

</td>
<td>

```eval_rst
.. mermaid::
   
   flowchart LR
      A(Node A)
      B([Node B])

      A -- points to --> B
      A --> C{{Node C}}

      classDef blue color:#0171c3;
      class B,C blue;
```

Mermaid is a charting tool for dynamically creating/modifying diagrams. Refer [here](https://mermaid-js.github.io/) for more Mermaid syntax.

</td></tr>
</table>

### 3.1 Use reStructuredText in `.md` files
You could embed reStructuredText into `.md` files through putting reStructuredText code into `eval_rst` code block. It is really useful when you want to use components such as sepcial boxes, tabs, cards, Mermaid diagrams, etc. in your `.md` file.
~~~md
```eval_rst
any contents in reStructuredText syntax
```

```eval_rst
.. note::
   
   This is a note box.

.. mermaid::
   
   flowchart LR
      A --> B
```
~~~

```eval_rst
.. important::

   Any contents inside ``eval_rst`` code block should follow the reStructuredText syntax.
```

## 4. Common components in `.md` files
<table class="table bigdl-documentation-guide-tables">
<tr><td>Headers</td>
<td>

```md
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

```md
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

Note that the number of spaces indented depends on the markup. That is, if we use '- '/'1. '/'10. ' for the list, the following contents belong to the list or the nested lists after it should be indented by 2/3/4 spaces.

</td></tr>
<tr><td>Code blocks</td>
<td>

~~~md
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

## 5. How to include Jupyter notebooks directly inside our documentation
If you want to include a Jupyter notebook into our documentation as an example, a tutorial, a how-to guide, etc., you could just put it anywhere inside [`BigDL/docs/readthedocs/source`](https://github.com/intel-analytics/BigDL/tree/main/docs/readthedocs/source) dictionary, and link it into `_toc.yml` file.

However, if you want to render a Jupyter notebook located out of `BigDL/docs/readthedocs/source` dictionary into our documentation, the case is a little bit complicated. To do this, you need to add a file with `.nblink` extension into `BigDL/docs/readthedocs/source` , and link the `.nblink` file into `_toc.yml`.

The `.nblink` file should have the following structure:
```json
{
    "path": "relative/path/to/the/notebook/you/want/to/include"
}
```

```eval_rst
.. seealso::

   You could find `here <https://github.com/intel-analytics/BigDL/blob/main/docs/readthedocs/source/doc/Nano/Howto/Training/PyTorchLightning/accelerate_pytorch_lightning_training_ipex.nblink>`_ for an example of ``.nblink`` usage inside our documentation.
```

### 5.1 How to hide a cell from rendering

If you want to hide a notebook markdown/code cell from rendering into our documentation, you could simply add `"nbsphinx": "hidden"` into the cell's `metadata`.

Here shows an examlpe of a markdown cell hidden from rendering:

```json
{
"cell_type": "markdown",
"metadata": {
   "nbsphinx": "hidden"
},
"source": [
   ...
]
}
```

```eval_rst
.. tip::

   You could simply open the notebook through text editor to edit the ``metadata`` of each cell.

.. note::

   Currently we could not hide the output/input code cell individually from rendering as they have the same ``metadata``. 
```

### 5.2 Note/Warning/Related Readings boxes
In convension, in the markdown cell of notebooks, we create note/warning/related reading boxes with the help of quote blocks and emoji:

```md
> 📝 **Note**
>
> This is a note box in notebooks.

> ⚠️ **Warning**
> 
> This is a warning box in notebooks.

> 📚 **Related Readings**
> 
> This is a related readings box in notebooks.

```