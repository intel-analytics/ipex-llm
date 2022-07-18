Performance tuning and analysis
===============================

.. highlight:: sh

This is the "best performance practices" document for Gramine performance
tuning and explanation of possible software and hardware performance bottlenecks
and limitations. In this document, we are highlighting only those manifest
options relevant for performance benchmarking. Refer to
:doc:`../manifest-syntax` for a full list.

*Note:* The below examples were run on a Mehlow machine. The performance numbers
in these examples should not be considered representative and serve only
illustration purposes.

Enabling per-thread and process-wide SGX stats
----------------------------------------------

See also :ref:`perf` below for installing ``perf``.

Enable statistics using ``sgx.enable_stats = true`` manifest option. Now your
graminized application correctly reports performance counters. This is useful
when using e.g. ``perf stat`` to collect performance statistics. This manifest
option also forces Gramine to dump SGX-related information on each
thread/process exit. Here is an example:

::

   LibOS/shim/test/regression$ perf stat gramine-sgx helloworld
   Hello world (helloworld)!
   ----- SGX stats for thread 87219 -----
   # of EENTERs:        224
   # of EEXITs:         192
   # of AEXs:           201
   # of sync signals:   32
   # of async signals:  0
   ----- Total SGX stats for process 87219 -----
   # of EENTERs:        224
   # of EEXITs:         192
   # of AEXs:           201
   # of sync signals:   32
   # of async signals:  0

   Performance counter stats for 'gramine-sgx helloworld':
        3,568,568,948      cycles
        1,072,072,581      instructions
          172,308,653      branches

How to read this output:

#. You can see that one thread was created, with Linux-host (actual) ``TID =
   87219``. This one thread belongs to one process, so they have the same ``TID
   = PID`` and the same statistics. If the test application would have e.g. two
   threads, then there would be two "SGX stats for thread" outputs, and their
   values would sum up to the values reported in the final "Total SGX stats for
   process".

#. There are about 200 EENTERs and EEXITs. These are mostly due to OCALLs:
   recall that every OCALL requires one EEXIT to exit the enclave and one EENTER
   to re-enter it again. We can conclude that there are about 200 OCALLs. Also,
   the number of EENTERs is slightly higher than of EEXITs: this is because of
   ECALLs.  Recall that Gramine performs one ECALL for the whole process and
   one ECALL per each new thread (and possibly some more ECALLs to reset threads
   and other bookkeeping). You can consider ECALLs as "no-return" in Gramine:
   they only contribute to EENTERs and do not increase EEXITs (in reality,
   ECALLs perform EEXITs but only after the stats are printed).

#. Why there are about 200 OCALLs in our trivial HelloWorld example? This is
   because even before HelloWorld's ``main()`` function starts, Gramine and
   Glibc initialize themselves. For example, Gramine must open and read the
   manifest file – this requires several OCALLs. Glibc may need to open and load
   shared libraries – this requires more OCALLs. In general, you may consider
   200 OCALLs as the cost of initialization in Gramine.

#. Why are there about 200 Asynchronous Exits (AEXs)? Most of these AEXs come
   from in-enclave exceptions and normal Linux interrupts. In particular,
   Linux’s scheduler interrupts each CPU every 4ms, or 250 times per second.
   Since our enclave workload runs for a fraction of a second, maybe 100 AEXs
   happen due to scheduler interrupts. Another 32 AEXs happen due to "sync
   signals" – in particular, SGX and Gramine trap-and-emulate the CPUID
   instruction. Each trap-and-emulate results in one AEX. Finally, the rest 100
   AEXs happen due to some other sources of interrupts: enclave page faults or
   network interrupts.

#. What are the 32 sync signals? These are synchronous signals forwarded by the
   host Linux to Gramine. Synchronous signals are SIGILL (invalid instruction),
   SIGFPE (invalid floating-point result), SIGSEGV (segmentation fault), etc.
   These are exceptions originating from Glibc/application execution. For
   example, these 32 sync signals are trap-and-emulate cases for the CPUID
   instruction, forbidden in SGX enclaves (we know this because we manually
   inspected this logic). In particular, Glibc initializes itself and checks for
   different CPU information, and thus invokes CPUID 32 times.

#. What are the 0 async signals? These are asynchronous signals forwarded by the
   host Linux to Gramine. These signals are SIGINT (interrupt), SIGCONT
   (continue), SIGKILL (the process is going to be killed), SIGCHLD (child
   process terminated) etc. None of these signals happened during HelloWorld
   execution.

#. Performance counters provide a lot of different information. Here we only
   show the snippet with number of cycles, instructions, and branches taken. By
   itself, this particular output is not interesting. In reality, performance
   counters should be compared against "golden runs" to deduce any interesting
   trends.

Effects of system calls / ocalls
--------------------------------

One of the main sources of overhead on modern SGX-enabled Intel processors is a
pure software one: enclavized applications must communicate with the outside
world, and for this communication they must perform system calls / OCALLs. Every
OCALL results in one EEXIT to exit from the enclave to the untrusted world and
then one EENTER to re-enter the enclave (unless you are using Exitless, see
below).  Moreover, OCALLs typically copy some data from the enclave to the
outside and vice versa – for example, network and file system OCALLs must copy
network packets and files to/from the enclave.

So, there are several sources of overhead that need to be understood with regard
to OCALLs:

#. ``OCALL = EEXIT + host processing + EENTER``. Recall that each EEXIT flushes
   the CPU caches and possibly invalidates Branch Predictors and TLBs.
   Similarly, EENTER performs many checks and requires hardware-internal
   synchronization of cores. Some studies show each EEXIT and EENTER cost around
   8,000 – 12,000 cycles (compare it with normal syscalls costing around 100
   cycles each). Note that the cost of EENTER/EEXIT depends on the CPU model,
   its firmware, applied microcode patches, and other platform characteristics.

#. OCALLs purge CPU caches. This means that after each OCALL, data that was
   cached in say L1 data cache is not there anymore. This effectively negates
   the effect of warm caches in the SGX environment.

#. Many OCALLs perform I/O: they copy data to/from the enclave. Copying is
   obligatory to prevent Time-of-check to time-of-use (TOCTOU) attacks and is
   dictated by the SGX design. This is an unavoidable tax. In I/O intensive
   workloads, the overhead of copying may constitute 15-50% over the baseline
   performance of the native application. For example, databases and web servers
   copy user requests inside the enclave and copy the results/web pages out. In
   another example, applications that manipulate files perform a lot of file
   system I/O, copying data blocks in and out of the enclave.

#. OCALLs generally correspond 1:1 to the system calls that the application
   performs, but not always. Typical system calls like ``read()``, ``write()``,
   ``recv()``, ``send()`` indeed correspond 1:1 to Gramine's OCALLs and thus
   introduce almost no overhead in the code path. However, some system calls are
   emulated in a more sophisticated way: e.g., Linux-specific ``epoll()`` is
   emulated via more generic ``poll()`` and this requires some additional logic.
   Fortunately, such calls are never a real bottleneck in Gramine because they
   are not on hot paths of applications. Probably the only exceptional system
   call is ``gettimeofday()`` – and only on older Intel CPUs (see below).

#. The ``gettimeofday()`` system call is special. On normal Linux, it is
   implemented via vDSO and a fast RDTSC instruction. Platforms older than
   Icelake typically forbid RDTSC inside an SGX enclave (this is a hardware
   limitation), and so ``gettimeofday()`` falls back to the expensive OCALL.
   Gramine is smart enough to identify whether the platform supports RDTSC
   inside enclaves, and uses the fast RDTSC logic to emulate ``gettimeofday()``.
   *Rule of thumb:* if you think that the bottleneck of your deployment is
   ``gettimeofday()``, move to a newer (Icelake) processor. If you cannot move
   to a newer platform, you are limited by SGX hardware (you can try to modify
   the application itself to issue less gettimeofday’s).

Exitless feature
----------------

Gramine supports the Exitless (or Switchless) feature – it trades off CPU cores
for faster OCALL execution. More specifically, with Exitless, enclave threads do
not exit the enclave on OCALLs but instead busy wait for untrusted helper
threads which perform OCALLs (system calls) on their behalf.  Untrusted helper
threads are created at Gramine start-up and burn CPU cycles busy waiting for
requests for OCALLs from enclave threads (untrusted helper threads periodically
sleep if there have been no OCALL requests for a long time to save some CPU
cycles).

Exitless is configured by ``sgx.rpc_thread_num = xyz``. By default, the Exitless
feature is disabled – all enclave threads perform an actual OCALL for each
system call and exit the enclave. The feature can be disabled by specifying
``sgx.rpc_thread_num = 0``.

You must decide how many untrusted helper RPC threads your application needs. A
rule of thumb: specify ``sgx.rpc_thread_num == sgx.thread_num``, i.e., the
number of untrusted RPC threads should be the same as the number of enclave
threads. For example, native Redis 6.0 uses 3-4 enclave threads during its
execution, plus Gramine uses another 1-2 helper enclave threads. So Redis
manifest has an over-approximation of this number: ``sgx.thread_num = 8``. Thus,
to correctly enable the Exitless feature, specify ``sgx.rpc_thread_num = 8``.
Here is an example:

::

   # exitless disabled: `sgx.thread_num = 8` and `sgx.rpc_thread_num = 0`
   CI-Examples/redis$ gramine-sgx redis-server --save '' --protected-mode no &
   CI-Examples/redis$ src/src/redis-benchmark -t set
   43010.75 requests per second

   # exitless enabled: `sgx.thread_num = 8` and `sgx.rpc_thread_num = 8`
   CI-Examples/redis$ gramine-sgx redis-server --save '' --protected-mode no &
   CI-Examples/redis$ src/src/redis-benchmark -t set
   68119.89 requests per second

As you can see, enabling the Exitless feature improves performance of Redis by
58%. This comes at a price: there are now 8 additional threads occupying
additional CPU cores (you can see these additional threads by running ``ps -Haux
| grep loader`` while Gramine is running). We recommend to use Exitless only for
single-threaded applications or if you care more about latency than throughput.

We also recommend to use core pinning via taskset or even isolating cores via
``isolcpus`` or disabling interrupts on cores via ``nohz_full``. It is also
beneficial to put all enclave threads on one set of cores (e.g., on first
hyper-threads if you have hyper-threading enabled on your platform) and all
untrusted RPC threads on another set of cores (e.g., on second hyper-threads).
In general, the classical performance-tuning strategies are applicable for
Gramine and Exitless multi-threaded workloads.

Optional CPU features (AVX, AVX512, MPX, PKRU, AMX)
---------------------------------------------------

SGX technology allows to specify which CPU features are required to run the SGX
enclave. Gramine "inherits" this and has the following manifest options:
``sgx.require_avx``, ``sgx.require_avx512``, ``sgx.require_mpx``,
``sgx.require_pkru``, ``sgx.require_amx``. By default, all of them are set to
``false`` – this means that SGX hardware will allow running the SGX enclave on
any system, whether the system has AVX/AVX512/MPX/PKRU/AMX features or not.

Gramine typically correctly identifies the features of the underlying platform
and propagates the information on AVX/AVX512/MPX/PKRU/AMX inside the enclave and
to the application. It is recommended to leave these manifest options as-is (set
to ``false``). However, we observed on some platforms that the graminized
application cannot detect these features and falls back to a slow
implementation. For example, some crypto libraries do not recognize AVX on the
platform and use very slow functions, leading to 10-100x overhead over native
(we still don't know the reason for this behavior). If you suspect this can be
your case, enable the features in the manifest, e.g., set ``sgx.require_avx =
true``.

For more information on SGX logic regarding optional CPU features, see the Intel
Software Developer Manual, Table 38-3 ("Layout of ATTRIBUTES Structure") under
the SGX section.

Multi-threaded workloads
------------------------

Gramine supports multi-threaded applications. Gramine implements many
optimizations and performance-relevant system calls related to multi-threading
and scheduling policies (e.g., ``set_schedaffinity()``).

Multi-process workloads
-----------------------

Gramine supports multi-process applications, i.e., applications that run as
several inter-dependent processes. Typical examples are bash scripts: one main
bash script spawns many additional processes to perform some operations.
Another typical example is Python: it usually spawns helper processes to obtain
system information. Finally, many applications are multi-process by design,
e.g., Nginx and Apache web servers spawn multiple worker processes.

For each new child, the parent Gramine process creates a new process with a new
Gramine instance and thus a new enclave. For example, if Nginx main process
creates 4 workers, then there will be 5 Gramine instances and 5 SGX enclaves:
one main Gramine process with its enclave and 4 child Gramine processes with 4
enclaves.

To create a new child process, Linux has the following system calls:
``fork()``/``vfork()`` and ``clone()``. All these interfaces copy the whole
memory of the parent process into the child, as well as all the resources like
opened files, network connections, etc. In a normal environment, this copying is
very fast because it uses the copy-on-write semantics. However, the SGX hardware
doesn't have the notions of copy-on-write  and sharing of memory. Therefore,
Gramine emulates ``fork/vfork/clone`` via the checkpoint-and-restore mechanism:
all enclave memory and resources of the parent process are serialized into one
blob of data, the blob is encrypted and sent to the child process. The child
process awaits this blob of data, receives it, decrypts it, and restores into
its own enclave memory. This is a much more expensive operation than
copy-on-write, therefore forking in Gramine is much slower than in native
Linux. Some studies report 1,000x overhead of forking over native.

Moreover, multi-process applications periodically need to communicate with each
other. For example, the Nginx parent process sends a signal to one of the worker
processes to inform that a new request is available for processing. All this
Inter-Process Communication (IPC) is transparently encrypted in Gramine.
Encryption by itself incurs 1-10% overhead. This means that a
communication-heavy multi-process application may experience significant
overheads.

To summarize, there are two sources of overhead for multi-process applications
in Gramine:

#. ``Fork()``, ``vfork()`` and ``clone()`` system calls are very expensive in
   Gramine and in SGX in general. This is because Intel SGX lacks the
   mechanisms for memory sharing and copy-on-write semantics. They are emulated
   via checkpoint-and-restore in Gramine.

#. Inter-Process Communication (IPC) is moderately expensive in Gramine because
   all IPC is transparently encrypted/decrypted using the TLS-PSK with AES-GCM
   crypto.

Choice of SGX machine
---------------------

Modern Icelake server machines remove many of the hardware bottlenecks of Intel
SGX. If you must use an older machine (Skylake, Caby Lake, Mehlow), you should
be aware that they have severe SGX-hardware limitations. In particular:

#. :term:`EPC` size. You can think of EPC as a physical cache (just like L3
   cache) for enclave pages. On older machines (before Icelake servers), EPC is
   only 128-256MB in size. This means that if the application has a working set
   size of more than 100-200MB, enclave pages will be evicted from EPC into RAM.
   Eviction of enclave pages (also called EPC swapping or paging) is a very
   expensive hardware operation. Some applications have a working set size of
   MBs/GBs of data, so performance will be significantly impaired.

   Note that modern Icelake servers have EPC size of up to 1TB and therefore
   they are not affected by EPC swapping. A simple way to verify the amount of
   EPC available on your machine is to execute Gramine's utility
   ``is-sgx-available``.

#. RDTSC/RDTSCP instructions. These instructions are forbidden to execute in an
   SGX enclave on older machines. Unfortunately, many applications and runtimes
   use these instructions frequently, assuming that they are always available.
   This leads to significant overheads when running such applications: Gramine
   treats each RDTSC instruction as trap-and-emulate, which is very expensive
   (enclave performs an AEX, Gramine enters the enclave, fixes RDTSC, exits the
   enclave, and re-enters it from the interrupted point). Solution: move to
   newer Intel processors that like Icelake which allow RDTSC inside the
   enclave.

#. CPUID and SYSCALL instructions. These instructions are forbidden to execute
   in an SGX enclave on all currently available machines. Fortunately,
   applications use these instructions typically only during initialization and
   never on hot paths. Gramine emulates CPUID and SYSCALL similarly to RDTSC,
   but since this happens very infrequently, it is not a realistic bottleneck.
   However, it is always advisable to verify that the application doesn’t rely
   on CPUID and SYSCALL too much. This is especially important for statically
   built applications that may rely on raw SYSCALL instructions instead of
   calling Glibc (recall that Gramine replaces native Glibc with our patched
   version that performs function calls inside Gramine instead of raw SYSCALL
   instructions and thus avoids this overhead).

#. CPU topology. The CPU topology may negatively affect performance of Gramine.
   For example, if the machine has several NUMA domains, it is important to
   restrict Gramine runs to only one NUMA domain, e.g., via the command
   ``numactl --cpunodebind=0 --membind=0``. Otherwise Gramine may spread
   enclave threads and enclave memory across several NUMA domains, which will
   lead to higher memory access latencies and overall worse performance.

Glibc malloc tuning
-------------------

Depending on the number of threads and the value of ``sgx.enclave_size``, you
might encounter pathological performance due to a combination of various factors.

Specifically, the default settings of glibc's ``malloc`` assume that virtual memory is
virtually unlimited, and, as an optimization, request a per-thread arena
of 64 MiB from the kernel when a thread first calls ``malloc``.

Due to the limitations of SGX v1, we must back each allocation with physical memory
immediately, which breaks the assumption that speculatively allocating 64 MiB is not
a big deal — when many threads are spawned (and call ``malloc``), the per-thread arenas
might consume a large portion of the memory reserved for the enclave.

When this happens, calls to ``malloc`` won't fail, as the allocator will
allocate a single page to serve the request instead. However, no attempt will
be made to make use of the rest of the page, wasting most of the memory.
Moreover, glibc will retry allocating the arena each time ``malloc`` gets
called, perhaps in a hope that the memory situation that prevented the previous
attempt from succeeding has since passed.

All together, this means that, unless ``64M * (application's thread count)`` fits
comfortably in ``sgx.enclave_size``, ``malloc`` will be much slower and much less
memory-efficient than it should be, on some of the threads involved, because each
call will now cause multiple relatively expensive calls to ``mmap``, and effectively
round up the request size to a multiple of 4096.

One way to solve this is to limit the number of threads that are allowed to have
their own arena. This can be done with either a call to ``mallopt``, or an environment
variable set in the manifest::

    # Only create one malloc arena.
    loader.env.MALLOC_ARENA_MAX = "1"

This does have its own performance implications, but the impact is much smaller
than the pathological behavior described above.

Another solution would be to set ``sgx.enclave_size`` to a much higher value,
to accomodate each thread creating its own arena. Do keep in mind, however,
that each process spawns its own enclave, so in a multi-process application,
the actual memory consumption will be a multiple of this setting. If the memory
consumption is not a problem for your usecase, you might observe better
performance with this approach than when limiting ``MALLOC_ARENA_MAX``.

Other considerations
--------------------

For performance testing, always use the non-debug versions of all software. In
particular, build Gramine in non-debug configuration
(``meson --buildtype=release``). Also build the application itself in non-debug
configuration (in the example Makefiles, simple ``make SGX=1`` defaults to
non-debug). Finally, disable the debug log of Gramine by specifying the manifest
option ``loader.log_level = "none"``.

There are several manifest options that may improve performance of some
workloads. The manifest options include:

- ``libos.check_invalid_pointers = false`` -- disable checks of invalid pointers
  on system call invocations. Most real-world applications never provide invalid
  arguments to system calls, so there is no need in additional checks.
- ``sgx.preheat_enclave = true`` -- pre-fault all enclave pages during enclave
  initialization. This shifts the overhead of page faults on non-present enclave
  pages from runtime to enclave startup time. Using this option makes sense only
  if the whole enclave memory fits into :term:`EPC`.

If your application periodically fails and complains about seemingly irrelevant
things, it may be due to insufficient enclave memory. Please try to increase
enclave size by tweaking ``sgx.enclave_size = "512M"``,
``sgx.enclave_size = "1G"``, ``sgx.enclave_size = "2G"``, and so on. If this
doesn't help, it could be due to insufficient stack size: in this case try to
increase ``sys.stack.size = "256K"``, ``sys.stack.size = "2M"``,
``sys.stack.size = "4M"`` and so on. Finally, if Gramine complains about
insufficient number of TCSs or threads, increase ``sgx.thread_num = 4``,
``sgx.thread_num = 8``, ``sgx.thread_num = 16``, and so on.

Do not forget about the cost of software encryption! Gramine transparently
encrypts many means of communication:

#. Inter-Process Communication (IPC) is encrypted via TLS-PSK. Regular pipes,
   FIFO pipes, UNIX domain sockets are all transparently encrypted.

#. Files mounted as ``type = "encrypted"`` are transparently encrypted/decrypted
   on each file access via SGX SDK Merkle-tree format.

#. ``Fork/vfork/clone`` all require to generate an encrypted checkpoint of the
   whole enclave memory, send it from parent process to the child, and decrypt
   it (all via TLS-PSK).

#. All SGX attestation, RA-TLS, and Secret Provisioning network communication is
   encrypted via TLS. Moreover, attestation depends on the internet speed and
   the remote party, so can also become a bottleneck.

Parsing the manifest can be another source of overhead. If you have a really
long manifest (several MBs in size), parsing such a manifest may significantly
deteriorate start-up performance. This is rarely a case, but keep manifests as
small as possible. Note that this overhead is due to our sub-optimal parser.
Once Gramine moves to a better manifest parser, this won't be an issue.

Finally, recall that by default Gramine doesn't propagate environment variables
into the SGX enclave. Thus, environment variables like ``OMP_NUM_THREADS`` and
``MKL_NUM_THREADS`` are not visible to the graminized application by default.
To propagate them into the enclave, either use the insecure manifest option
``loader.insecure__use_host_env = true`` (don't use this in production!) or
specify them explicitly in the manifest via
``loader.env.OMP_NUM_THREADS = "8"``. Also, it is always better to specify such
environment variables explicitly because a graminized application may determine
the number of available CPUs incorrectly.

.. _perf:

Profiling with ``perf``
-----------------------

This section describes how to use `perf
<https://perf.wiki.kernel.org/index.php/Main_Page>`__, a powerful Linux
profiling tool.

Installing ``perf`` provided by your distribution
"""""""""""""""""""""""""""""""""""""""""""""""""

Under Ubuntu:

#. Install ``linux-tools-common``.
#. Run ``perf``. It will complain about not having a kernel-specific package,
   such as ``linux-tools-4-15.0-122-generic``.
#. Install the kernel-specific package.

The above might not work if you have a custom kernel. In that case, you might
want to use the distribution-provided version anyway (install
``linux-tools-generic`` and use ``/usr/lib/linux-tools-<VERSION>/perf``), but it
might not be fully compatible with your kernel. It might be better to build
your own.

You might also want to compile your own ``perf`` to make use of libraries that
the default version is not compiled against.

Building your own ``perf``
""""""""""""""""""""""""""

#. Download the kernel: run ``uname -r`` to check your kernel version, then
   clone the right branch::

       git clone --single-branch --branch linux-5.4.y \
           https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git

#. Go to ``linux/tools/perf`` and run ``make``.

#. Check the beginning of the output. It will display warnings about missing
   libraries, and suggest how to install them.

   Install the missing ones, depending on the features you need. You will need
   at least ``libdw-dev`` and ``libunwind-dev`` to get proper symbols and stack
   trace. ``libslang2-dev`` is also nice, as it will enable a terminal UI for
   some commands.

#. Run ``make`` again and verify that the necessary features have been
   enabled. (You can also run ``ldd perf`` to check which shared libraries it
   uses).

#. Install somewhere, e.g. ``sudo make install DESTDIR=/usr/local``.

Recording samples with ``perf record``
""""""""""""""""""""""""""""""""""""""

To record (saves ``perf.data``)::

    perf record gramine-direct application

To view the report for ``perf.data``::

    perf report

This is useful in non-SGX mode. Unfortunately, in SGX mode, it will not account
correctly for the code inside the enclave.

Some useful options for recording (``perf record``):

* ``--call-graph dwarf``: collect information about callers
* ``-F 50``: collect 50 (or any other number) of samples per second,
  can be useful to reduce overhead and file size
* ``-e cpu-clock``: sample the ``cpu-clock`` event, which will be triggered also
  inside enclave (as opposed to the default ``cpu-cycles`` event). Unfortunately
  such events will be counted towards ``async_exit_pointer`` instead of
  functions executing inside enclave (but see also :ref:`sgx-profile`).

Some useful options for displaying the report (``perf report``):

* ``--no-children``: sort based on "self time", i.e. time spent in a given
  function excluding its children (the default is to sort by total time spent in
  a function).

Further reading
"""""""""""""""

* `Perf Wiki <https://perf.wiki.kernel.org/index.php/Main_Page>`__
* `Linux perf examples - Brendan Gregg
  <http://www.brendangregg.com/perf.html>`__
* Man pages: ``man perf record``, ``man perf report`` etc.

.. _sgx-profile:

SGX profiling
-------------

There is support for profiling the code inside the SGX enclave. Here is how to
use it:

#. Compile Gramine with ``-Dsgx=enabled --buildtype=debugoptimized``.

   You can also use ``--buildtype=debug``, but ``--buildtype=debugoptimized``
   (optimizations enabled) makes Gramine performance more similar to release
   build.

#. Add ``sgx.profile.enable = "main"`` to manifest (to collect data for the main
   process), or ``sgx.profile.enable = "all"`` (to collect data for all
   processes).

#. (Add ``sgx.profile.with_stack = true`` for call chain information.)

#. Run your application. It should say something like ``Profile data written to
   sgx-perf.data`` on process exit (in case of ``sgx.profile.enable = "all"``,
   multiple files will be written).

#. Run ``perf report -i <data file>`` (see :ref:`perf` above).

*Note*: The accuracy of this tool is unclear (though we had positive experiences
using the tool so far). The SGX profiling works by measuring the value of
instruction pointer on each asynchronous enclave exit (AEX), which happen on
Linux scheduler interrupts, as well as other events such as page faults. While
we attempt to measure time (and not only count occurences), the results might be
inaccurate.

.. _sgx-profile-ocall:

OCALL profiling
"""""""""""""""

It's also possible to discover what OCALLs are being executed, which should help
attribute the EEXIT numbers given by ``sgx.enable_stats``. There are two ways to
do that:

* Use ``sgx.profile.mode = "ocall_inner"`` and ``sgx.profile.with_stack =
  1``. This will give you a report on what enclave code is causing the OCALLs
  (best viewed with ``perf report --no-children``).

  The ``with_stack`` option is important: without it, the report will only show
  the last function before enclave exit, which is usually the same regardless of
  which OCALL we're executing.

* Use ``sgx.profile.mode = "ocall_outer"``. This will give you a report on what
  outer PAL code is handling the OCALLs (``sgx_ocall_open``, ``sgx_ocall_write``
  etc.)

**Warning**: The report for OCALL modes should be interpreted in term of *number
of OCALLs*, not time spent in them. The profiler records a sample every time an
OCALL is executed, and ``perf report`` displays percentages based on the number
of samples.


.. _vtune-sgx-profiling:

Profiling SGX hotspots with Intel VTune Profiler
----------------------------------------------------

This section describes how to use `VTune
<https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/introduction.html>`__
profiler to find SGX hotspots.

Installing VTune
""""""""""""""""""""

Please download and install Intel VTune Profiler from `here
<https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/installation.html>`__.

Collecting SGX hotspots and viewing the report
""""""""""""""""""""""""""""""""""""""""""""""

#. Compile Gramine with
   ``-Dvtune=enabled -Dvtune_sdk_path=<VTune SDK install path>``.

   If ``vtune_sdk_path`` is not provided, Gramine will use the default VTune
   installation path.

#. Add ``sgx.vtune_profile = true`` and ``sgx.debug = true`` to the manifest.

#. Run your application under VTune.

   ``vtune -collect sgx-hotspots -result-dir results -- gramine-sgx <workload>``

   It will output the data collected to a directory ``results``.

#. To view the report, run
   ``vtune -report hotspots -r <vtune data collection output directory>`` or use
   Intel VTune Profiler GUI application.


Other useful tools for profiling
--------------------------------

* ``strace -c`` will display Linux system call statistics
* Valgrind (with `Callgrind
  <https://valgrind.org/docs/manual/cl-manual.html>`__) unfortunately doesn't
  work, see `issue #1919
  <https://github.com/gramineproject/graphene/issues/1919>`__ for discussion.
