GRAPHENEDIR ?= /graphene
SGX_SIGNER_KEY ?= /graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem
ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

THIS_DIR ?= /ppml/trusted-big-data-ml
JDK_HOME ?= /opt/jdk8

G_JAVA_XMX ?= 2G
G_SGX_SIZE ?= 65536M
G_SGX_THREAD_NUM ?= 256

ifeq ($(DEBUG),1)
GRAPHENE_LOG_LEVEL = debug
else
GRAPHENE_LOG_LEVEL = error
endif

.PHONY: all
all: bash.manifest
ifeq ($(SGX),1)
all: bash.manifest.sgx bash.sig bash.token
endif

bash.manifest: bash.manifest.template
	graphene-manifest \
                -Dsgx_enclave_size=$(G_SGX_SIZE) \
                -Dlog_level=$(GRAPHENE_LOG_LEVEL) \
                -Dexecdir=$(shell dirname $(shell which bash)) \
                -Darch_libdir=$(ARCH_LIBDIR) \
                -Djdk_home=$(JDK_HOME) \
                $< >$@

bash.manifest.sgx: bash.manifest
	@test -s $(SGX_SIGNER_KEY) || \
            { echo "SGX signer private key was not found, please specify SGX_SIGNER_KEY!"; exit 1; }
	graphene-sgx-sign \
                --key $(SGX_SIGNER_KEY) \
                --libpal $(GRAPHENEDIR)/Runtime/libpal-Linux-SGX.so \
                --manifest bash.manifest \
                --output $@

bash.sig: bash.manifest.sgx

bash.token: bash.sig
	graphene-sgx-get-token --output bash.token --sig bash.sig

ifeq ($(SGX),)
GRAPHENE = graphene-direct
else
GRAPHENE = graphene-sgx
endif

.PHONY: regression
regression: all
	@mkdir -p scripts/testdir

	$(GRAPHENE) ./bash -c "ls" > OUTPUT
	@grep -q "Makefile" OUTPUT && echo "[ Success 1/7 ]"
	@rm OUTPUT

	$(GRAPHENE) ./bash -c "cd scripts && bash bash_test.sh 1" > OUTPUT
	@grep -q "hello 1" OUTPUT      && echo "[ Success 2/7 ]"
	@grep -q "createdfile" OUTPUT  && echo "[ Success 3/7 ]"
	@grep -q "somefile" OUTPUT     && echo "[ Success 4/7 ]"
	@grep -q "current date" OUTPUT && echo "[ Success 5/7 ]"
	@rm OUTPUT

	$(GRAPHENE) ./bash -c "cd scripts && bash bash_test.sh 3" > OUTPUT
	@grep -q "hello 3" OUTPUT      && echo "[ Success 6/7 ]"
	@rm OUTPUT

	$(GRAPHENE) ./bash -c "readlink /proc/self/exe" > OUTPUT
	@grep -qE "^(/usr)?/bin/readlink" OUTPUT && echo "[ Success 7/7 ]"
	@rm OUTPUT

	@rm -rf scripts/testdir


.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig OUTPUT scripts/testdir/*

.PHONY: distclean
distclean: clean
