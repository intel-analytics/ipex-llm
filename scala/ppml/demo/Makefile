GRAPHENEDIR ?= /graphene
SGX_SIGNER_KEY ?= /graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem
ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

THIS_DIR ?= /opt
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
        /graphene/meson_build_output/bin/graphene-manifest \
                -Dsgx_enclave_size=$(G_SGX_SIZE) \
                -Dlog_level=$(GRAPHENE_LOG_LEVEL) \
                -Dexecdir=$(shell dirname $(shell which bash)) \
                -Darch_libdir=$(ARCH_LIBDIR) \
                -Djdk_home=$(JDK_HOME) \
                $< >$@

bash.manifest.sgx: bash.manifest
        @test -s $(SGX_SIGNER_KEY) || \
            { echo "SGX signer private key was not found, please specify SGX_SIGNER_KEY!"; exit 1; }
        /graphene/meson_build_output/bin/graphene-sgx-sign \
                --key $(SGX_SIGNER_KEY) \
                --libpal $(GRAPHENEDIR)/Runtime/libpal-Linux-SGX.so \
                --manifest bash.manifest \
                --output $@

bash.sig: bash.manifest.sgx

bash.token: bash.sig
        /graphene/meson_build_output/bin/graphene-sgx-get-token --output bash.token --sig bash.sig

ifeq ($(SGX),)
GRAPHENE = graphene-direct
else
GRAPHENE = graphene-sgx
endif

.PHONY: clean
clean:
        $(RM) *.manifest *.manifest.sgx *.token *.sig OUTPUT scripts/testdir/*

.PHONY: distclean
distclean: clean
ss