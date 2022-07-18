Implementing new system call
============================

.. highlight:: c

1. Define interface of system call and add it to system call table
------------------------------------------------------------------

For example, assume we are implementing :manpage:`sched_setaffinity(2)`. You
must add the prototype of the function implementing it to :file:`shim_table.h`::

   long shim_do_sched_setaffinity(pid_t pid, unsigned int len, unsigned long* user_mask_ptr);

Note that we use the following naming convetion: ``shim_do_`` followed by
an actual syscall name. Additionally this function should return ``long``.
Now you need to add an appropriate entry in the syscalls table in
:file:`shim_table-$(ARCH).c`::

    [__NR_sched_setaffinity] = (shim_fp)shim_do_sched_setaffinity

2. Implement system call
------------------------

You can add the function body of ``shim_do_sched_setaffinity`` in a new source
file or any existing source file in :file:`LibOS/shim/src/sys`.

For example, in :file:`LibOS/shim/src/sys/shim_sched.c`::

   long shim_do_sched_setaffinity(pid_t pid, unsigned int len, unsigned long* user_mask_ptr) {
      /* code for implementing the semantics of sched_setaffinity */
   }

3. Add new PAL Calls (optional)
-------------------------------

The concept of Gramine library OS is to keep the PAL interface as simple as
possible. So, you should not add new PAL calls if the features can be fully
implemented inside the library OS using the existing PAL calls. However,
sometimes the OS features needed involve low-level operations inside the host OS
and cannot be emulated inside the library OS. Therefore, you may have to add
a |~| few new PAL calls to the existing interface.

To add a |~| new PAL call, first modify :file:`Pal/include/pal/pal.h`. Define
the PAL call::

   bool DkThreadSetCPUAffinity(PAL_NUM cpu_num, PAL_IDX* cpu_indexes);

The naming convention of a |~| PAL call is to start functions with the ``Dk``
prefix, followed by a comprehensive name describing the purpose of the PAL
call.

4. Export new PAL calls from PAL binaries (optional)
----------------------------------------------------

For each directory in :file:`PAL/host/`, there is a :file:`pal.map` file. This
file lists all the symbols accessible to the library OS. The new PAL call needs
to be listed here in order to be used by your system call implementation.

5. Implement new PAL calls (optional)
-------------------------------------

.. todo::

   (Not finished...)
