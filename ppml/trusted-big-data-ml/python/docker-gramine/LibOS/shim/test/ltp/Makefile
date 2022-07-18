
ROOTDIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

SRCDIR = $(ROOTDIR)/src
BUILDDIR = $(ROOTDIR)/build
INSTALLDIR = $(ROOTDIR)/install
TESTCASEDIR = $(INSTALLDIR)/testcases/bin

clean-extra += clean-build

.PHONY: all
all: $(INSTALLDIR)/INSTALL_SUCCESS manifests etc/nsswitch.conf etc/passwd

$(SRCDIR)/Makefile:
	$(error "$(SRCDIR) is empty. Please run `git submodule update --init $(SRCDIR)` or download the LTP source code (https://github.com/linux-test-project/ltp) into $(SRCDIR).")

$(SRCDIR)/configure: $(SRCDIR)/Makefile
	$(MAKE) -C $(SRCDIR) autotools

.SECONDARY: $(BUILDDIR)/BUILD_SUCCESS
$(BUILDDIR)/BUILD_SUCCESS: $(SRCDIR)/configure
	# Out-of-tree build steps were taken from ltp/INSTALL.
	# The instructions below assume that SRCDIR and BUILDDIR are absolute.
	mkdir -p $(BUILDDIR)
	# Kernel module tests are not meaningful for our LibOS and building them causes troubles on
	# incompatible host kernels.
	cd $(BUILDDIR) && $(SRCDIR)/configure --without-modules --prefix $(INSTALLDIR)
	$(MAKE) -C $(BUILDDIR) -f $(SRCDIR)/Makefile top_srcdir=$(SRCDIR) top_builddir=$(BUILDDIR) all
	touch $@

.SECONDARY: $(INSTALLDIR)/INSTALL_SUCCESS
$(INSTALLDIR)/INSTALL_SUCCESS: $(BUILDDIR)/BUILD_SUCCESS
	$(MAKE) -C $(BUILDDIR) -f $(SRCDIR)/Makefile top_srcdir=$(SRCDIR) top_builddir=$(BUILDDIR) SKIP_IDCHECK=1 install
	ln -sf $(abspath Makefile.testcases) $(TESTCASEDIR)/Makefile
	touch $@

.PHONY: manifests
manifests: $(INSTALLDIR)/INSTALL_SUCCESS
	gramine-test build

etc/nsswitch.conf:
	mkdir -p etc
	printf "passwd: compat\ngroup: compat\nshadow: compat\nhosts: files\n" > $@

etc/passwd:
	mkdir -p etc
	printf "root:x:0:0:root:/root:/bin/bash\nnobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n" > $@

ifeq ($(SGX),"1")
CFG = ltp.cfg ltp-sgx.cfg ltp-bug-1075.cfg
else
CFG = ltp.cfg
endif

.PHONY: regression
regression: manifests
	LTP_CONFIG="$(CFG)" python3 -m pytest -v

.PHONY: clean
clean:
	if test -f $(INSTALLDIR)/INSTALL_SUCCESS; then gramine-test clean; fi
	$(MAKE) -C $(SRCDIR) clean
	$(RM) -r \
		$(BUILDDIR) \
		$(INSTALLDIR) \
		ltp*.xml \
		etc/ \
		.pytest_cache \
		__pycache__
