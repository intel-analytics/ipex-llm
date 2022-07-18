Coding style guidelines
=======================

This document describes coding conventions and formatting styles we use in
Gramine. All newly commited code must conform to them to pass a |~| review.

Automatic reformatting
----------------------

To make formatting easier we've added an integration with
:program:`clang-format` (currently only for C |~| code). You must install
appropriate package from your distribution to use it. For Ubuntu 18.04 you can
setup it this way:

.. code-block:: sh

   sudo apt-get install clang-format

Usage: (assuming you've configured your build into ``build`` directory)

.. code-block:: sh

   ninja -C build/ clang-format

This :command:`make` target **reformats all source files in-place**, so we
recommend you first commit them (or add to `git index
<https://hackernoon.com/understanding-git-index-4821a0765cf>`__ with
:command:`git add -A`), reformat and then verify reformatting results using
:command:`git diff` (or :command:`git diff --cached` if you used :command:`git
add`).

.. warning::

   Because of bugs in clang-format and its questionable reformats in many places
   (seems it deals with C++ much better than with C) it's intended only as a |~|
   helper tool. Adding it to git pre-commit hooks is definitely a |~| bad idea,
   at least currently.

C
-

We use a style derived (and slightly modified) from `Google C++ Styleguide
<https://google.github.io/styleguide/cppguide.html>`__.

Code formatting
^^^^^^^^^^^^^^^

.. note::

   See our :file:`.clang-format` config for precise rules.

#. Indentation: 4 spaces per level.

#. Maximal line length: 100 characters.

#. Brace placement::

      void f() {
          if (a && b) {
              something();
          }
      }

#. ``if-else`` formatting::

      if (x == y) {
          ...
      } else if (x > y) {
          ...
      } else {
          ...
      }

#. Asterisks (``*``) should be placed on the left, with the type. Multiple
   pointer declarations in one line are disallowed. Example::

      int* pointer;
      int* another_pointer;
      int non_pointer_a, non_pointer_b, non_pointer_c;

#. Function call/declaration folding: aligned to a matching parenthesis.
   Required only if the one-line version would exceed the line length limit.
   Examples::

      int many_args(int something_looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong,
                    int also_looooooong,
                    int c);
      ...
      many_args(some_looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong_calculations,
                many_args(123,
                          also_looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong,
                          789),
                many_args(1, 2, 3));

#. ``if``, ``else``, ``do``, ``for``, ``while``, ``switch`` and ``union`` should
   be followed by a space.

#. Includes should be grouped and then sorted lexicographically. Groups should
   be separated using a |~| single empty line.

   Groups:

   #. Matching :file:`.h` header for :file:`.c` files.
   #. Standard library headers.
   #. Non-standard headers not included in Gramine's repository (e.g. from
      external dependencies, like :file:`curl.h`).
   #. Gramine's headers.

#. Assignments may be aligned when assigning some structurized data (e.g. struct
   members). Example::

      int some_int = 0;
      bool asdf = true;
      file->size      = 123;
      file->full_path = "/asdf/ghjkl";
      file->perms     = PERM_rw_r__r__;

Conventions and high-level style
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Variable and function names should be sane and easy to understand (example:
   ``nofpts`` is bad, ``points_cnt`` is ok). The names ``i``, ``j``, ``k`` etc.
   should be limited to integers used as array indexes.

#. All non-static function interfaces should be documented in comments
   (especially pointer ownerships). Same for public macros.

#. Prefer readable code and meaningful variable/function names to explaining
   implementation details in comments within a |~| function. Only tricky or
   unintuitive code should be commented.

#. Inline comments should be separated from code (or macros) with one space.

#. Magic numbers (e.g. buffer sizes) shouldn’t be hardcoded in the
   implementation. Use ``#define``.

#. Naming:

   #. Macros and global constants should be ``NAMED_THIS_WAY``.
   #. Functions, structures and variables should be ``named_this_way``.
   #. Global variables should be prefixed with ``g_`` (e.g. ``g_thread_list``).
   #. "size" always means size in bytes, "length" (or "count") means the number
      of elements (e.g. in an array, or characters in a C-string, excluding the
      terminating null byte).

#. Types:

    #. All in-memory sizes and array indexes should be stored using ``size_t``.
    #. All file offsets and sizes should be stored using ``file_off_t``.
    #. In general, C99 types should be used where possible (although some code
       is "grandfathered" in, it should also be changed as time allows).

#. ``goto`` may be used only for error handling.

#. `Yoda conditions <https://en.wikipedia.org/wiki/Yoda_conditions>`__
   (e.g. ``if (42 == x)``) or any other similar constructions are not allowed.

#. Prefer ``sizeof(instance)`` to ``sizeof(type)``, it’s less error-prone.

Python
------

#. Executable Python scripts must use the shebang with the hardcoded path to
   system Python (e.g., ``#!/usr/bin/python3``). This is required because custom
   Python installations ("custom" meaning not provided by distro) lead to a
   problem where packages installed via e.g. ``apt install`` are not available
   to this custom Python. If Python scripts would use the ``#!/usr/bin/env
   python3`` shebang, Gramine would not be able to locate system-wide-installed
   Python packages.

   Since Gramine currently supports only Debian/Ubuntu and CentOS/RHEL/Fedora
   distros, the shebang must always be ``#!/usr/bin/python3``.

Meson
-----

#. 4-space indent, no tabs. Wrap lines at ~80-100 columns except for unbreakable
   things like URLs.

#. First argument to target functions (``shared_library``, ``executable``,
   ``custom_target``, ...) should be on the same line as opening paren. All
   other arguments should be on next lines, aligned to 4-space indent.

   Arguments to other functions should either be all on the same line, or there
   should be no argument on the same line as opening paren, and arguments should
   be in following lines, indented by 4 spaces.

#. Otherwise, whitespace should generally follow PEP8 instead of meson suggested
   style (i.e., no space inside parens, no space before ``:``).

#. No changing (overwriting) variables in different :file:`meson.build` than it
   was defined in. If you really need to do this, create a temporary variable
   in subdir and use it in the parent :file:`meson.build`. You can check
   ``libos_sources_arch`` in :file:`LibOS/shim/src/meson.build` for example
   usage of this pattern (appending arch-specific source files to a list).

#. Variables named ``_prog`` refer to things obtained from ``find_program()``.
   Auxiliary commands should reside in ``Scripts/``, and the variable name is
   tied to the script name (see :file:`meson.build` there). The scripts should
   be written in Python except for things that clearly benefit from being
   written in ``sh``.
