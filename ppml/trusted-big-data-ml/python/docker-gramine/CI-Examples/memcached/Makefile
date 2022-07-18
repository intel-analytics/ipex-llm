ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

SRCDIR = src
MEMCACHED_SRC ?= memcached-1.5.21.tar.gz
MEMCACHED_MIRRORS ?= \
    https://memcached.org/files \
    https://packages.gramineproject.io/distfiles

MEMCACHED_SHA256 ?= e3d10c06db755b220f43d26d3b68d15ebf737a69c7663529b504ab047efe92f4

UBUNTU_VER = $(shell lsb_release --short --id)$(shell lsb_release --short --release)

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

.PHONY: all
all: memcached memcached.manifest
ifeq ($(SGX),1)
all: memcached.manifest.sgx memcached.sig memcached.token
endif

$(SRCDIR)/configure:
	../common_tools/download --output memcached.tar.gz --sha256 $(MEMCACHED_SHA256) \
		$(foreach mirror,$(MEMCACHED_MIRRORS),--url $(mirror)/$(MEMCACHED_SRC))
	mkdir $(SRCDIR)
	tar -C $(SRCDIR) --strip-components=1 -xf memcached.tar.gz

$(SRCDIR)/memcached: $(SRCDIR)/configure
	cd $(SRCDIR) && ./configure
	$(MAKE) -C $(SRCDIR)

memcached.manifest: memcached.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		$< > $@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# see the helloworld example for details on this workaround.
memcached.manifest.sgx memcached.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: memcached.manifest memcached
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

memcached.token: memcached.sig
	gramine-sgx-get-token \
		--output memcached.token --sig memcached.sig

# for simplicity, copy memcached executable into our root directory
memcached: $(SRCDIR)/memcached
	cp $< $@

.PHONY: start-native-server
start-native-server: all
	./memcached

ifeq ($(SGX),)
GRAMINE = gramine-direct
else
GRAMINE = gramine-sgx
endif

.PHONY: start-gramine-server
start-gramine-server: all
	$(GRAMINE) memcached -u nobody

.PHONY: clean
clean:
	$(RM) *.token *.sig *.manifest.sgx *.manifest memcached .lck

.PHONY: distclean
distclean: clean
	$(RM) -r $(SRCDIR) memcached.tar.gz
