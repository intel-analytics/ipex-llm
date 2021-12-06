GRAPHENEDIR ?= ../..
SGX_SIGNER_KEY ?= $(GRAPHENEDIR)/Pal/src/host/Linux-SGX/signer/enclave-key.pem
ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

G_JAVA_XMX ?= 2G
G_SGX_SIZE ?= 8G
G_LOG_LEVEL ?= debug
G_SGX_THREAD_NUM ?= 256

THIS_DIR ?= /ppml/trusted-big-data-ml
JDK_HOME ?= /opt/jdk8
SPARK_HOME ?= $(THIS_DIR)/work/spark-3.1.2
WORK_DIR ?= $(THIS_DIR)/work
SPARK_USER ?= root
SPARK_LOCAL_IP ?= 127.0.0.1

ifeq ($(DEBUG),1)
GRAPHENE_LOG_LEVEL = debug
else
GRAPHENE_LOG_LEVEL = error
endif

.PHONY: all
all: bash.manifest | pal_loader
ifeq ($(SGX),1)
all: bash.token
endif

include $(GRAPHENEDIR)/Scripts/Makefile.configs
PYTHONVERSION = python3.6
include /graphene/Scripts/Makefile.python

bash.manifest: bash.manifest.template
	sed -e 's|$$(GRAPHENEDIR)|'"$(GRAPHENEDIR)"'|g' \
             -e 's|$$(GRAPHENE_LOG_LEVEL)|'"$(GRAPHENE_LOG_LEVEL)"'|g' \
             -e 's|$$(EXECPATH)|'"$(shell which bash)"'|g' \
             -e 's|$$(EXECDIR)|'"$(shell dirname $(shell which bash))"'|g' \
             -e 's|$$(ARCH_LIBDIR)|'"$(ARCH_LIBDIR)"'|g' \
             -e 's|$$(JDK_HOME)|'"$(JDK_HOME)"'|g' \
             -e 's|$$(SPARK_HOME)|'"$(SPARK_HOME)"'|g' \
             -e 's|$$(SPARK_LOCAL_IP)|'"$(SPARK_LOCAL_IP)"'|g' \
             -e 's|$$(SPARK_USER)|'"$(SPARK_USER)"'|g' \
             -e 's|$$(WORK_DIR)|'"$(WORK_DIR)"'|g' \
             -e 's|$$(PYTHONDISTHOME)|'"$(PYTHONDISTHOME)"'|g' \
             -e 's|$$(PYTHONHOME)|'"$(PYTHONHOME)"'|g' \
             -e 's|$$(PYTHONEXEC)|'"$(PYTHONEXEC)"'|g' \
             -e 's|$$(G_SGX_SIZE)|'"$(G_SGX_SIZE)"'|g' \
             -e 's|$$(G_LOG_LEVEL)|'"$(G_LOG_LEVEL)"'|g' \
             $< > $@

bash.manifest.sgx: bash.manifest
	$(GRAPHENEDIR)/Pal/src/host/Linux-SGX/signer/pal-sgx-sign \
                -libpal $(GRAPHENEDIR)/Runtime/libpal-Linux-SGX.so \
                -key $(SGX_SIGNER_KEY) \
                -manifest $< -output $@

bash.sig: bash.manifest.sgx

bash.token: bash.sig
	$(GRAPHENEDIR)/Pal/src/host/Linux-SGX/signer/pal-sgx-get-token \
                -output bash.token -sig bash.sig
pal_loader:
	ln -s $(GRAPHENEDIR)/Runtime/pal_loader $@

.PHONY: regression
regression: all
	@mkdir -p scripts/testdir

	./pal_loader ./bash -c "ls" > OUTPUT
	@grep -q "Makefile" OUTPUT && echo "[ Success 1/6 ]"
	@rm OUTPUT

	./pal_loader ./bash -c "cd scripts && bash bash_test.sh 1" > OUTPUT
	@grep -q "hello 1" OUTPUT      && echo "[ Success 2/6 ]"
	@grep -q "createdfile" OUTPUT  && echo "[ Success 3/6 ]"
	@grep -q "somefile" OUTPUT     && echo "[ Success 4/6 ]"
	@grep -q "current date" OUTPUT && echo "[ Success 5/6 ]"
	@rm OUTPUT

	./pal_loader ./bash -c "cd scripts && bash bash_test.sh 3" > OUTPUT
	@grep -q "hello 3" OUTPUT      && echo "[ Success 6/6 ]"
	@rm OUTPUT

	@rm -rf scripts/testdir


.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig trusted-libs pal_loader OUTPUT scripts/testdir/*

.PHONY: distclean
distclean: clean
