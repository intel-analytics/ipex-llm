Development setup
=================

For contributors, we strongly suggest using the following configuration
according to your editors.

Emacs configuration
-------------------

No change needed. See :file:`.dir-locals.el`.

Vim configuration
-----------------

Please add the following script to the end of your :file:`~/.vimrc`,
or place in :file:`~/.vim/after/ftplugin/c.vim` if you have other plugins.

.. code-block:: vim

   let dirname = expand('%:p:h')
   let giturl = system('cd '.dirname.'; git config --get remote.origin.url 2>/dev/null')
   if giturl =~ 'gramineproject/gramine'
      set textwidth=100 tabstop=4 softtabstop=4 shiftwidth=4 expandtab
   endif

   au BufRead,BufNewFile *.rst imap <A-Space> <Space>\|~\|<Space>
   au BufRead,BufNewFile *.rst set textwidth=80

.. warning::

   Due to security concerns, we do not suggest using Vim modelines or
   :file:`.exrc`.
