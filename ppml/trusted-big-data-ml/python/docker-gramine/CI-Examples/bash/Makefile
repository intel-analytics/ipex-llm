ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

.PHONY: all
all: bash.manifest
ifeq ($(SGX),1)
all: bash.manifest.sgx bash.sig bash.token
endif

bash.manifest: manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Dexecdir=$(shell dirname $(shell which bash)) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		$< >$@

bash.manifest.sgx: bash.manifest
	gramine-sgx-sign \
		--manifest bash.manifest \
		--output $@

bash.sig: bash.manifest.sgx

bash.token: bash.sig
	gramine-sgx-get-token --output bash.token --sig bash.sig

ifeq ($(SGX),)
GRAMINE = gramine-direct
else
GRAMINE = gramine-sgx
endif

.PHONY: regression
regression: all
	@mkdir -p scripts/testdir

	$(GRAMINE) ./bash -c "ls" > OUTPUT
	@grep -q "Makefile" OUTPUT && echo "[ Success 1/7 ]"
	@rm OUTPUT

	$(GRAMINE) ./bash -c "cd scripts && bash bash_test.sh 1" > OUTPUT
	@grep -q "hello 1" OUTPUT      && echo "[ Success 2/7 ]"
	@grep -q "createdfile" OUTPUT  && echo "[ Success 3/7 ]"
	@grep -q "somefile" OUTPUT     && echo "[ Success 4/7 ]"
	@grep -q "current date" OUTPUT && echo "[ Success 5/7 ]"
	@rm OUTPUT

	$(GRAMINE) ./bash -c "cd scripts && bash bash_test.sh 3" > OUTPUT
	@grep -q "hello 3" OUTPUT      && echo "[ Success 6/7 ]"
	@rm OUTPUT

	$(GRAMINE) ./bash -c "readlink /proc/self/exe" > OUTPUT
	@grep -qE "^(/usr)?/bin/readlink" OUTPUT && echo "[ Success 7/7 ]"
	@rm OUTPUT

	@rm -rf scripts/testdir


.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig OUTPUT scripts/testdir/*

.PHONY: distclean
distclean: clean
