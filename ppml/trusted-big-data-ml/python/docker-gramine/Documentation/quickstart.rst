Quick start
===========

.. highlight:: sh

Prerequisites
-------------

Gramine without SGX has no special requirements.

Gramine with SGX support requires several features from your system:

- the FSGSBASE feature of recent processors must be enabled in the Linux kernel;
- the Intel SGX driver must be built in the Linux kernel;
- Intel SGX SDK/PSW and (optionally) Intel DCAP must be installed.

If your system doesn't meet these requirements, please refer to more detailed
descriptions in :doc:`devel/building`.

We supply a tool :doc:`manpages/is-sgx-available`, which you can use to check
your hardware and system. It's installed together with the respective gramine
package (see below).

Install Gramine
---------------

On Ubuntu 18.04 or 20.04 (for 18.04, in :file:`intel-sgx.list`, replace
``focal`` with ``bionic``)::

   sudo curl -fsSLo /usr/share/keyrings/gramine-keyring.gpg https://packages.gramineproject.io/gramine-keyring.gpg
   echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/gramine-keyring.gpg] https://packages.gramineproject.io/ stable main' | sudo tee /etc/apt/sources.list.d/gramine.list

   curl -fsSL https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | sudo apt-key add -
   echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list
   # (if you're on Ubuntu 18.04, remember to write "bionic" instead of "focal")

   sudo apt-get update

   sudo apt-get install gramine      # for 5.11+ upstream, in-kernel driver
   sudo apt-get install gramine-oot  # for out-of-tree SDK driver
   sudo apt-get install gramine-dcap # for out-of-tree DCAP driver

On RHEL-8-like distribution (like AlmaLinux 8, CentOS 8, Rocky Linux 8, ...)::

   sudo curl -fsSLo /etc/yum.repos.d/gramine.repo https://packages.gramineproject.io/rpm/gramine.repo
   sudo dnf install gramine          # only the default, distro-provided kernel is supported

Prepare a signing key
---------------------

Only for SGX, and if you haven't already::

   gramine-sgx-gen-private-key

This command generates an |~| RSA 3072 key suitable for signing SGX enclaves and
stores it in :file:`{HOME}/.config/gramine/enclave-key.pem`. This key needs to
be protected and should not be disclosed to anyone.

Run sample application
----------------------

Core Gramine repository contains several sample applications. Thus, to test
Gramine installation, we clone the Gramine repo:

.. parsed-literal::

   git clone --depth 1 \https://github.com/gramineproject/gramine.git |stable-checkout|

We don't want to build Gramine (it is already installed on the system). Instead,
we want to build and run the HelloWorld example. To build the HelloWorld
application, we need the ``gcc`` compiler and the ``make`` build system::

   sudo apt-get install gcc make  # for Ubuntu distribution
   sudo dnf install gcc make      # for RHEL-8-like distribution

Go to the HelloWorld example directory::

   cd gramine/CI-Examples/helloworld

Build and run without SGX::

   make
   gramine-direct helloworld

Build and run with SGX::

   make SGX=1
   gramine-sgx helloworld

Other sample applications
-------------------------

We prepared and tested several applications to demonstrate Gramine usability.
These applications can be found in the :file:`CI-Examples` directory in the
repository, each containing a short README with instructions how to test it. We
recommend starting with a simpler, thoroughly documented example of Redis, to
understand manifest options and features of Gramine.

Additional sample configurations for applications enabled in Gramine can be
found in a separate repository https://github.com/gramineproject/examples.

Please note that these sample applications are tested on Ubuntu 18.04 and 20.04.
Most of these applications are also known to run correctly on
Fedora/RHEL/CentOS, but with caveats. One caveat is that Makefiles should be
invoked with ``ARCH_LIBDIR=/lib64 make``. Another caveat is that applications
that rely on specific versions/builds of Glibc may break (our GCC example is
known to work only on Ubuntu).

glibc vs musl
-------------

Most of the examples we provide use GNU C Library (glibc). If your application
is built against musl libc, you can pass ``'musl'`` to
:py:func:`gramine.runtimedir()` when generating the manifest from a template,
which will mount musl libc (instead of the default glibc).
