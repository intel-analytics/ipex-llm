ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

.PHONY: all
all: sqlite3.manifest
ifeq ($(SGX),1)
all: sqlite3.manifest.sgx sqlite3.sig sqlite3.token
endif

sqlite3.manifest: manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Dexecdir=$(shell dirname $(shell which sqlite3)) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		$< >$@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# see the helloworld example for details on this workaround.
sqlite3.manifest.sgx sqlite3.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: sqlite3.manifest
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

sqlite3.token: sqlite3.sig
	gramine-sgx-get-token --output sqlite3.token --sig sqlite3.sig

ifeq ($(SGX),)
GRAMINE = gramine-direct
else
GRAMINE = gramine-sgx
endif

.PHONY: regression
regression: all
	@rm -f scripts/testdir/*

	$(GRAMINE) sqlite3 scripts/testdir/test.db < scripts/create.sql
	@echo "[ Success 1/3 ]"

	$(GRAMINE) sqlite3 scripts/testdir/test.db < scripts/update.sql
	@echo "[ Success 2/3 ]"

	$(GRAMINE) sqlite3 scripts/testdir/test.db < scripts/select.sql > OUTPUT
	diff OUTPUT scripts/select.txt
	@echo "[ Success 3/3 ]"

.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig OUTPUT scripts/testdir/*

.PHONY: distclean
distclean: clean
