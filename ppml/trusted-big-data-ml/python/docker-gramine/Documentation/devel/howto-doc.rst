.. _howto-doc:

How to write documentation
==========================

.. highlight:: rst

Gramine uses `Sphinx`_ to generate the documentation in HTML form. The
documentation is hosted on Read The Docs at https://gramine.readthedocs.io/.
Documentation is generally written as `reStructuredText`_ files which are placed
in ``Documentation/`` directory. See `Sphinx' reST primer`_ for short
introduction into syntax.

For code written in |~| C, we use `Doxygen`_ and `Breathe`_, which is
a |~| Sphinx' plugin for including Doxygen documentation. Documentation of
C |~| language API should be written as Doxygen comments (see `Doxygen manual`_)
and then included in one of the ``.rst`` files (with appropriate description)
using one of the `Breathe directives`_, like ``.. doxygenfunction::`` or ``..
doxygenstruct::``.

The documentation should be written with ``html`` builder of Sphinx in mind. The
:file:`manpages/` subdirectory also targets ``manpage`` builder. Other builders
(like ``latex``) may be considered in the future, but for now their output is
not published.

.. note::

   A |~| note about terminology:

   ``html``, ``latex`` and ``manpage``, and also others, are Sphinx "builders":
   https://www.sphinx-doc.org/en/master/man/sphinx-build.html#cmdoption-sphinx-build-b.
   Sphinx can output many different formats, some of them have overlapping usage
   (both ``html`` and ``latex`` usually output full handbook, the difference is
   screen vs print), some are specialized (``manpage`` processes only selected
   documents for man; those documents may or may not be also used by other
   builders).

   When launched through ``make`` (like ``make -C Documentation html``), this
   becomes "target" in Make terminology.

Building documentation
----------------------

To build documentation, change directory to ``Documentation``, install prerequisites, and use
``make``, specifying the appropriate target. The documentation is built with python3; if you have
similar packages in python2, it may create problems; we recommend removing any similar packages in
python2. Similarly, the documentation requires version 1.8 of sphinx.

The output is in the ``_build`` directory:

.. code-block:: sh

   # change directory to Documentation
   cd Documentation

   # install prerequisites
   sudo apt-get install doxygen
   python3 -m pip install -r requirements.txt

   # build targets "html" and "man"
   make html man

   # example: view html output
   firefox _build/html/index.html

   # example: view man output
   man _build/man/gramine-manifest.1

Preferred reST style
--------------------

(This is adapted from `Python's style guide`_).

- Use 3-space tab in ``.rst`` files to align the indentation with reST explicit
  markup, which begins with two dots and a |~| space.

- Wrap the paragraphs at 80th character. But don't wrap verbatim text like logs
  and use applicable style when wrapping code examples (see
  :doc:`coding-style`).

- For headers, use Python convention for header hierarchy:

  #. ``#`` with overline,
  #. ``*`` with overline,
  #. ``=``,
  #. ``-``,
  #. ``^``,
  #. ``"``.

  Example::

     ###################################
     Very top level header (in TOC etc.)
     ###################################

     *******************
     Less than top level
     *******************

     Per-file header
     ===============

     Section header
     --------------

     Subsection header
     ^^^^^^^^^^^^^^^^^

     Subsubsection header
     """"""""""""""""""""

  This means most documents use only ``=`` and ``-`` adornments.

  .. tip::

     For vim users:
        you can enter the ``-`` underlines using the key combination
        ``yypVr-`` and the other adornments with similar combinations.

     For Emacs users:
        Read more at https://docutils.sourceforge.io/docs/user/emacs.html.

- Use ``|~|`` to insert non-breaking space. This should be added after
  one-letter words and where otherwise appropriate::

      This is a |~| function.

  This substitution is added to all documents processed by Sphinx. For files
  processed also by other software (like ``README.rst``, which is both rendered
  by GitHub and included in ``index.rst``), use ``|nbsp|`` after adding this
  substitution yourself::

      .. |nbsp| unicode:: 0xa0
         :trim:

      This is a |nbsp| README.

Documentation of the code should be organized into files by logical concepts,
as they fit into programmer's mind. Ideally, this should match the source files,
if those files were organized correctly in the first place, but the reality may
be different. In case of doubt, place them as they fit the narrative of the
document, not as they are placed in the source files.

Documents should be grouped by general areas and presented using
``.. toctree::`` directive in :file:`index.rst` file. This causes them to be
included in TOC in the main document and also in sidebar on RTD.

Preferred Doxygen style
-----------------------

#. Prefer Qt-style ``/*!`` and ``\param``:

   .. Note that the snippet below is wrapped to 106 chars per line. This is
      because it quotes C code (wrapped to 100), and the quote is itself
      indented in reST.

   .. code-block:: c

      /*!
       * \brief Sum two integers.
       *
       * \param first   First addend.
       * \param second  Second addend.
       *
       * \returns Sum of the arguments. Sometimes a longer description is needed, then it should be
       *          wrapped and aligned like this.
       */
      int foo(int first, int second) {
          return first + second;
      }

   ::

      There is a |~| very special function :c:func:`foo`:

      .. doxygenfunction:: foo

      It's an example function, but is documented!

#. In reST, do not use ``autodoxygen`` directives, and especially do not use
   ``.. doxygenfile::``, because documentation should be written as prose, not
   a |~| coredump. Write an explanation, how the things go together and place
   the ``.. doxygenfunction::`` directives where aproppriate.

#. You can use ``\rst`` and ``\endrst`` to write reST in Doxygen comments:

   .. code-block:: c

      /*!
       * \brief An example function
       *
       * \rst
       * .. note::
       *
       *    This works!
       * \endrst
       */

Further reading
---------------

- `Four kinds of documentation`_
  (`HN thread <https://news.ycombinator.com/item?id=21289832>`__)
- `The Hitchhiker's Guide to Documentation`_ divided by audience (role in the
  project), with references to good real-world examples

.. _reStructuredText: https://en.wikipedia.org/wiki/ReStructuredText
.. _Sphinx: https://www.sphinx-doc.org/
.. _Sphinx' reST primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _Doxygen: http://www.doxygen.nl/
.. _Doxygen manual: http://www.doxygen.nl/manual/docblocks.html
.. _Breathe: https://breathe.readthedocs.io/en/latest/
.. _Breathe directives: https://breathe.readthedocs.io/en/latest/directives.html
.. _Python's style guide: https://devguide.python.org/documenting/#style-guide
.. _Four kinds of documentation: https://www.divio.com/blog/documentation/
.. _The Hitchhiker's Guide to Documentation: https://docs-guide.readthedocs.io/en/latest/>
