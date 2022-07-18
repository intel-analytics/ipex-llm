Introduction to SGX
===================

.. highlight:: sh

Gramine project uses :term:`SGX` to securely run software. SGX is
a |~| complicated topic, which may be hard to learn, because the documentation
is scattered through official/reference documentation, blogposts and academic
papers. This page is an attempt to curate a |~| dossier of available reading
material.

SGX is an umbrella name of *technology* that comprises several parts:

- CPU/platform *hardware features*: the new instruction set, new
  microarchitecture with the :term:`PRM` (:term:`EPC`) memory region and some
  new MSRs and some new logic in the MMU and so on;
- the SGX :term:`Remote Attestation` *infrastructure*, online services provided
  by Intel and/or third parties (see :term:`DCAP`);
- :term:`SDK` and assorted *software*.

SGX is still being developed. The current (March 2020) version of CPU features
is referred to as "SGX1" or simply "SGX" and is more or less finalized. All
new/changed instructions from original SGX are informally referred to as
":term:`SGX2`".

Features which might be considered part of SGX2:

- :term:`EDMM` (Enclave Dynamic Memory Management) is part of SGX2
- :term:`FLC` (Flexible Launch Control), not strictly part of SGX2, but was not
  part of original SGX hardware either

As of now there is hardware support (on a |~| limited set of CPUs) for FLC and
(on an even more limited set of CPUs) SGX2/EDMM. Most of the literature
available (especially introduction-level) concerns original SGX1 only.

Introductory reading
--------------------

- Quarkslab's two-part "Overview of Intel SGX":

  - `Part 1, SGX Internals (Quarkslab)
    <https://blog.quarkslab.com/overview-of-intel-sgx-part-1-sgx-internals.html>`__
  - `Part 2, SGX Externals (Quarkslab)
    <https://blog.quarkslab.com/overview-of-intel-sgx-part-2-sgx-externals.html>`__

- `MIT's deep dive in SGX architecture <https://eprint.iacr.org/2016/086>`__.

- Intel's whitepapers:

  - `Innovative Technology for CPU Based Attestation and Sealing
    <https://software.intel.com/en-us/articles/innovative-technology-for-cpu-based-attestation-and-sealing>`__
  - `Innovative Instructions and Software Model for Isolated Execution
    <https://software.intel.com/en-us/articles/innovative-instructions-and-software-model-for-isolated-execution>`__
  - `Using Innovative Instructions to Create Trustworthy Software Solutions [PDF]
    <https://software.intel.com/sites/default/files/article/413938/hasp-2013-innovative-instructions-for-trusted-solutions.pdf>`__
  - `Slides from ISCA 2015 <https://sgxisca.weebly.com/>`__
    (`actual slides [PDF] <https://software.intel.com/sites/default/files/332680-002.pdf>`__)

- `Hardware compatibility list (unofficial) <https://github.com/ayeks/SGX-hardware>`__

Official Documentation
----------------------

- `Intel® 64 and IA-32 Architectures Software Developer's Manual Volume 3D:
  System Programming Guide, Part 4
  <https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3d-part-4-manual.pdf>`__
- `SDK for Linux <https://01.org/intel-software-guard-extensions/downloads>`__
  (download of both the binaries and the documentation)

Academic Research
-----------------

- `Intel's collection of academic papers
  <https://software.intel.com/en-us/sgx/documentation/academic-research>`__,
  likely the most comprehensive list of references

Installation Instructions
-------------------------

.. todo:: TBD

Linux kernel drivers
^^^^^^^^^^^^^^^^^^^^

For historical reasons, there are three SGX drivers currently (January 2021):

- https://github.com/intel/linux-sgx-driver -- old one, does not support DCAP,
  deprecated

- https://github.com/intel/SGXDataCenterAttestationPrimitives/tree/master/driver
  -- new one, out-of-tree, supports both non-DCAP software infrastructure (with
  old EPID remote-attestation technique) and the new DCAP (with new ECDSA and
  more "normal" PKI infrastructure).

- SGX support was upstreamed to the Linux mainline starting from 5.11.
  It currently supports only DCAP attestation. The driver is accessible through
  /dev/sgx_enclave and /dev/sgx_provision.

  The following udev rules are recommended for users to access the SGX node::

    groupadd -r sgx
    gpasswd -a USERNAME sgx
    groupadd -r sgx_prv
    gpasswd -a USERNAME sgx_prv
    cat > /etc/udev/rules.d/65-gramine-sgx.rules << EOF
      SUBSYSTEM=="misc",KERNEL=="sgx_enclave",MODE="0660",GROUP="sgx"
      SUBSYSTEM=="misc",KERNEL=="sgx_provision",MODE="0660",GROUP="sgx_prv"
      EOF
    udevadm trigger

  Also it will not require :term:`IAS` and kernel maintainers consider
  non-writable :term:`FLC` MSRs as non-functional SGX:
  https://lore.kernel.org/lkml/20191223094614.GB16710@zn.tnic/

The chronicle of kernel patchset:

v1 (2016-04-25)
   https://lore.kernel.org/lkml/1461605698-12385-1-git-send-email-jarkko.sakkinen@linux.intel.com/
v2
   ?
v3
   ?
v4 (2017-10-16)
   https://lore.kernel.org/lkml/20171016191855.16964-1-jarkko.sakkinen@linux.intel.com/
v5 (2017-11-13)
   https://lore.kernel.org/lkml/20171113194528.28557-1-jarkko.sakkinen@linux.intel.com/
v6 (2017-11-25)
   https://lore.kernel.org/lkml/20171125193132.24321-1-jarkko.sakkinen@linux.intel.com/
v7 (2017-12-07)
   https://lore.kernel.org/lkml/20171207015614.7914-1-jarkko.sakkinen@linux.intel.com/
v8 (2017-12-15)
   https://lore.kernel.org/lkml/20171215202936.28226-1-jarkko.sakkinen@linux.intel.com/
v9 (2017-12-16)
   https://lore.kernel.org/lkml/20171216162200.20243-1-jarkko.sakkinen@linux.intel.com/
v10 (2017-12-24)
   https://lore.kernel.org/lkml/20171224195854.2291-1-jarkko.sakkinen@linux.intel.com/
v11 (2018-06-08)
   https://lore.kernel.org/lkml/20180608171216.26521-1-jarkko.sakkinen@linux.intel.com/
v12 (2018-07-03)
   https://lore.kernel.org/lkml/20180703182118.15024-1-jarkko.sakkinen@linux.intel.com/
v13 (2018-08-27)
   https://lore.kernel.org/lkml/20180827185507.17087-1-jarkko.sakkinen@linux.intel.com/
v14 (2018-09-25)
   https://lore.kernel.org/lkml/20180925130845.9962-1-jarkko.sakkinen@linux.intel.com/
v15 (2018-11-03)
   https://lore.kernel.org/lkml/20181102231320.29164-1-jarkko.sakkinen@linux.intel.com/
v16 (2018-11-06)
   https://lore.kernel.org/lkml/20181106134758.10572-1-jarkko.sakkinen@linux.intel.com/
v17 (2018-11-16)
   https://lore.kernel.org/lkml/20181116010412.23967-2-jarkko.sakkinen@linux.intel.com/
v18 (2018-12-22)
   https://lore.kernel.org/linux-sgx/20181221231134.6011-1-jarkko.sakkinen@linux.intel.com/
v19 (2019-03-20)
   https://lore.kernel.org/lkml/20190320162119.4469-1-jarkko.sakkinen@linux.intel.com/
v20 (2019-04-17)
   https://lore.kernel.org/lkml/20190417103938.7762-1-jarkko.sakkinen@linux.intel.com/
v21 (2019-07-13)
   https://lore.kernel.org/lkml/20190713170804.2340-1-jarkko.sakkinen@linux.intel.com/
v22 (2019-09-03)
   https://lore.kernel.org/lkml/20190903142655.21943-1-jarkko.sakkinen@linux.intel.com/
v23 (2019-10-28)
   https://lore.kernel.org/lkml/20191028210324.12475-1-jarkko.sakkinen@linux.intel.com/
v24 (2019-11-30)
   https://lore.kernel.org/lkml/20191129231326.18076-1-jarkko.sakkinen@linux.intel.com/
v25 (2020-02-04)
   https://lore.kernel.org/lkml/20200204060545.31729-1-jarkko.sakkinen@linux.intel.com/
v26 (2020-02-09)
   https://lore.kernel.org/lkml/20200209212609.7928-1-jarkko.sakkinen@linux.intel.com/
v27 (2020-02-23)
   https://lore.kernel.org/lkml/20200223172559.6912-1-jarkko.sakkinen@linux.intel.com/
v28 (2020-04-04)
   https://lore.kernel.org/lkml/20200303233609.713348-1-jarkko.sakkinen@linux.intel.com/
v29 (2020-04-22)
   https://lore.kernel.org/lkml/20200421215316.56503-1-jarkko.sakkinen@linux.intel.com/
v30 (2020-05-15)
   https://lore.kernel.org/lkml/20200515004410.723949-1-jarkko.sakkinen@linux.intel.com/

SGX terminology
---------------

.. keep this sorted by full (not abbreviated) terms, leaving out generic terms
   like "Intel" and "SGX"

.. glossary::

   Architectural Enclaves
   AE

      Architectural Enclaves (AEs) are a |~| set of "system" enclaves concerned
      with starting and attesting other enclaves. Intel provides reference
      implementations of these enclaves, though other companies may write their
      own implementations.

      .. seealso::

         :term:`Provisioning Enclave`

         :term:`Launch Enclave`

         :term:`Quoting Enclave`

   AEP
      .. todo:: TBD

   Architectural Enclave Service Manager
   AESM

      The Architectural Enclave Service Manager is responsible for providing SGX
      applications with access to the :term:`Architectural Enclaves`. It consists
      of the Architectural Enclave Service Manager Daemon, which hosts the enclaves,
      and a component of the SGX SDK, which communicates with the daemon over a Unix
      socket with the fixed path :file:`/var/run/aesmd/aesm.sock`.

   AEX
      .. todo:: TBD

   Attestation

      Attestation is a mechanism to prove the trustworthiness of the SGX enclave
      to a local or remote party. More specifically, SGX attestation proves that
      the enclave runs on a real hardware in an up-to-date TEE with the expected
      initial state. There are two types of the attestation:
      :term:`Local Attestation` and :term:`Remote Attestation`. For local
      attestation, the attesting SGX enclave collects attestation evidence in
      the form of an :term:`SGX Report` using the EREPORT hardware instruction.
      For remote attestation, the attesting SGX enclave collects attestation
      evidence in the form of an :term:`SGX Quote` using the :term:`Quoting
      Enclave` (and the :term:`Provisioning Enclave` if required). The enclave
      then may send the collected attestation evidence to the local or remote
      party, which will verify the evidence and confirm the correctness of the
      attesting enclave. After this, the local or remote party trusts the
      enclave and may establish a secure channel with the enclave and send
      secrets to it.

      .. seealso::

         :doc:`attestation`

         :term:`Local Attestation`

         :term:`Remote Attestation`

   Data Center Attestation Primitives
   DCAP

      A |~| software infrastructure provided by Intel as a reference
      implementation for the new ECDSA/:term:`PCS`-based remote attestation.
      Relies on the :term:`Flexible Launch Control` hardware feature. In
      principle this is a |~| special version of :term:`SDK`/:term:`PSW` that
      has a |~| reference launch enclave and is backed by the DCAP-enabled SGX
      driver.

      This allows for launching enclaves with Intel's remote infrastructure
      only involved in the initial setup. Naturally however, this requires
      deployment of own infrastructure, so is operationally more complicated.
      Therefore it is intended for server environments (where you control all
      the machines).

      .. seealso::

         Orientation Guide
            https://download.01.org/intel-sgx/dcap-1.0.1/docs/Intel_SGX_DCAP_ECDSA_Orientation.pdf

         :term:`EPID`
            A |~| way to launch enclaves with Intel's infrastructure, intended
            for client machines.

   Enclave
      .. todo:: TBD

   Enclave Dynamic Memory Management
   EDMM
      A |~| hardware feature of :term:`SGX2`, allows dynamic memory allocation,
      which in turn allows dynamic thread creation.

   Enclave Page Cache
   EPC

      .. todo:: TBD

   Enclave Page Cache Map
   EPCM

      .. todo:: TBD

   Enhanced Privacy Identification
   Enhanced Privacy Identifier
   EPID

      EPID is the attestation protocol originally shipped with SGX. Unlike
      :term:`DCAP`, a |~| remote verifier making use of the EPID protocol needs
      to contact the :term:`Intel Attestation Service` each time it wishes
      to attest an |~| enclave.

      Contrary to DCAP, EPID may be understood as "opinionated", with most
      moving parts fixed and tied to services provided by Intel. This is
      intended for client enclaves and deprecated for server environments.

      EPID attestation can operate in two modes: *fully-anonymous (unlinkable)
      quotes* and *pseudonymous (linkable) quotes*.  Unlike fully-anonymous
      quotes, pseudonymous quotes include an |~| identifier dependent on the
      identity of the CPU and the developer of the enclave being quoted, which
      allows determining whether two instances of your enclave are running on
      the same hardware or not.

      If your security model depends on enforcing that the identifiers are
      different (e.g. because you want to prevent sybil attacks), keep in mind
      that the enclave host can generate a new identity by performing an
      epoch reset. The previous identity will then become inaccessible, though.

      The attestation mode being used can be chosen by the application enclave,
      but it must match what was chosen when generating the :term:`SPID`.

      .. seealso::

         :term:`DCAP`
            A way to launch enclaves without relying on the Intel's
            infrastructure.

         :term:`SPID`
            An identifier one can obtain from Intel, required to make use of EPID
            attestation.

   Flexible Launch Control
   FLC

      Hardware (CPU) feature that allows substituting :term:`Launch Enclave` for
      one not signed by Intel. A |~| change in SGX's EINIT logic to not require
      the EINITTOKEN from the Intel-based Launch Enclave. An |~| MSR, which can
      be locked at boot time, keeps the hash of the public key of the
      "launching" entity.

      With FLC, :term:`Launch Enclave` can be written by other companies (other
      than Intel) and must be signed with the key corresponding to the one
      locked in the MSR (a |~| reference Launch Enclave simply allows all
      enclaves to run). The MSR can also stay unlocked and then it can be
      modified at run-time by the VMM or the OS kernel.

      Support for FLC can be detected using ``CPUID`` instruction, as
      ``CPUID.07H:ECX.SGX_LC[bit 30] == 1`` (SDM vol. 2A calls this "SGX Launch
      Control").

      .. seealso::

         https://software.intel.com/en-us/blogs/2018/12/09/an-update-on-3rd-party-attestation
            Announcement

         :term:`DCAP`

   Launch Enclave
   LE

      .. todo:: TBD

      .. seealso::

         :term:`Architectural Enclaves`

   Local Attestation

      In local attestation, the attesting SGX enclave collects attestation
      evidence in the form of an :term:`SGX Report` using the EREPORT hardware
      instruction. This form of attestation is used to send the attestation
      evidence to a local party (on the same physical machine).

      .. seealso::

         :doc:`attestation`

   Intel Attestation Service
   IAS

      Internet service provided by Intel for "old" :term:`EPID`-based remote
      attestation. Enclaves send SGX quotes to the client/verifier who will
      forward them to IAS to check their validity.

      .. seealso::

         :term:`PCS`
            Provisioning Certification Service, another Internet service
            provided by Intel.

   Memory Encryption Engine
   MEE

      .. todo:: TBD

   OCALL

      .. todo:: TBD

   SGX Platform Software
   PSW

      Software infrastructure provided by Intel with all special
      :term:`Architectural Enclaves` (:term:`Provisioning Enclave`,
      :term:`Quoting Enclave`, :term:`Launch Enclave`). This mainly refers to
      the "old" EPID/IAS-based remote attestation.

   Processor Reserved Memory
   PRM

      .. todo:: TBD

   Provisioning Enclave
   PE

      One of the Architectural Enclaves of the Intel SGX software
      infrastructure. It is part of the :term:`SGX Platform Software`. The
      Provisioning Enclave is used in :term:`EPID` based remote attestation.
      This enclave communicates with the Intel Provisioning Service
      (:term:`IPS`) to perform EPID provisioning. The result of this
      provisioning procedure is the private EPID key securely accessed by the
      Provisioning Enclave. This procedure happens only during the first
      deployment of the SGX machine (or, in rare cases, to provision a new EPID
      key after TCB upgrade). The main user of the Provisioning Enclave is the
      :term:`Quoting Enclave`.

      .. seealso::

         :term:`Architectural Enclaves`

   Provisioning Certification Enclave
   PCE

      One of the Architectural Enclaves of the Intel SGX software
      infrastructure. It is part of the :term:`SGX Platform Software` and
      :term:`DCAP`. The Provisioning Certification Enclave is used in
      :term:`DCAP` based remote attestation.  This enclave communicates with the
      Intel Provisioning Certification Service (:term:`PCS`) to perform DCAP
      provisioning. The result of this provisioning procedure is the DCAP/ECDSA
      attestation collateral (mainly the X.509 certificate chains rooted in a
      well-known Intel certificate and Certificate Revocation Lists). This
      procedure happens during the first deployment of the SGX machine and then
      periodically to refresh the cached attestation collateral. Typically, to
      reduce the dependency on PCS, a cloud service provider introduces an
      intermediate caching service (Provisioning Certification Caching Service,
      or PCCS) that stores all the attestation collateral obtained from Intel.
      The main user of the Provisioning Certification Enclave is the
      :term:`Quoting Enclave`.

      .. seealso::

         :term:`Architectural Enclaves`

   Intel Provisioning Service
   IPS

      Internet service provided by Intel for EPID-based remote attestation.
      This service provides the corresponding EPID key to the Provisioning
      Enclave on a remote SGX machine.

   Intel Provisioning Certification Service
   PCS

      New internet service provided by Intel for new ECDSA-based remote
      attestation. Enclave provider creates its own internal Attestation Service
      where it caches PKI collateral from Intel's PCS, and the verifier gets the
      certificate chain from the enclave provider to check validity.

      .. seealso::

         :term:`IAS`
            Intel Attestation Service, another Internet service.

   Quoting Enclave
   QE

      One of the Architectural Enclaves of the Intel SGX software
      infrastructure. It is part of the :term:`SGX Platform Software`. The
      Quoting Enclave receives an :term:`SGX Report` and produces a
      corresponding :term:`SGX Quote`. The identity of the Quoting Enclave is
      publicly known (it signer, its measurement and its attributes) and is
      vetted by public companies such as Intel (in the form of the certificate
      chain ending in a publicly known root certificate of the company).

      .. seealso::

         :term:`Architectural Enclaves`

   Remote Attestation

      In remote attestation, the attesting SGX enclave collects attestation
      evidence in the form of an :term:`SGX Quote` using the :term:`Quoting
      Enclave` (and the :term:`Provisioning Enclave` if required). This form of
      attestation is used to send the attestation evidence to a remote party
      (not on the same physical machine).

      .. seealso::

         :doc:`attestation`

   Intel SGX Software Development Kit
   Intel SGX SDK
   SGX SDK
   SDK

      In the context of :term:`SGX`, this means a |~| specific piece of software
      supplied by Intel which helps people write enclaves packed into ``.so``
      files to be accessible like normal libraries (at least on Linux).
      Available together with a |~| kernel module and documentation.

   SGX Enclave Control Structure
   SECS

      .. todo:: TBD

   SGX Quote

      The SGX quote is the proof of trustworthiness of the enclave and is used
      during :term:`Remote Attestation`. The attesting enclave generates the
      enclave-specific :term:`SGX Report`, sends the request to the
      :term:`Quoting Enclave` using :term:`Local Attestation`, and the Quoting
      Enclave returns back the SGX quote with the SGX report embedded in it. The
      resulting SGX quote contains the enclave's measurement, attributes and
      other security-relevant fields, and is tied to the identity of the
      :term:`Quoting Enclave` to prove its authenticity. The obtained SGX quote
      may be later sent to the verifying remote party, which examines the SGX
      quote and gains trust in the remote enclave.

   SGX Report

      The SGX report is a data structure that contains the enclave's measurement,
      signer identity, attributes and a user-defined 64B string. The SGX report
      is generated using the ``EREPORT`` hardware instruction. It is used during
      :term:`Local Attestation`. The SGX report is embedded into the
      :term:`SGX Quote`.

   SGX2

      This refers to all new SGX instructions and other hardware features that
      were introduced after the release of the original SGX1.

      Encompasses at least :term:`EDMM`, but is still work in progress.

   Service Provider ID
   SPID

      An identifier provided by Intel, used together with an |~| :term:`EPID`
      API key to authenticate to the :term:`Intel Attestation Service`. You can
      obtain an |~| SPID through Intel's `Trusted Services Portal
      <https://api.portal.trustedservices.intel.com/EPID-attestation>`_.

      See :term:`EPID` for a |~| description of the difference between *linkable*
      and *unlinkable* quotes.

   State Save Area
   SSA

      .. todo:: TBD

   Security Version Number
   SVN

      .. todo:: TBD

   Trusted Execution Environment
   TEE

      A Trusted Execution Environment (TEE) is an environment where the code
      executed and the data accessed are isolated and protected in terms of
      confidentiality (no one has access to the data except the code running
      inside the TEE) and integrity (no one can change the code and its
      behavior).

   Trusted Computing Base
   TCB

      In context of :term:`SGX` this has the usual meaning: the set of all
      components that are critical to security. Any vulnerability in TCB
      compromises security. Any problem outside TCB is not a |~| vulnerability,
      i.e. |~| should not compromise security.

      In context of Gramine there is also a |~| different meaning
      (:term:`Thread Control Block`). Those two should not be confused.

   Thread Control Structure
   TCS

      .. todo:: TBD
