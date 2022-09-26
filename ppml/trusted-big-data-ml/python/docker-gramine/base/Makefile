SGX_SIGNER_KEY ?= /root/.config/gramine/enclave-key.pem
ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

G_JAVA_XMX ?= "2G"
G_SGX_SIZE ?= "16G"
G_LOG_LEVEL ?= all
G_SGX_THREAD_NUM ?= 1024

THIS_DIR ?= /ppml/trusted-big-data-ml
WORK_DIR ?= $(THIS_DIR)/work
JDK_HOME ?= /opt/jdk8
PYTHON_HOME ?= /usr/lib/python3.8
SPARK_HOME ?= $(THIS_DIR)/work/spark-3.1.2
SPARK_USER ?= root
SPARK_LOCAL_IP ?= 127.0.0.1


.PHONY: all
all: bash.manifest
ifeq ($(SGX),1)
all: bash.manifest.sgx bash.sig bash.token
endif

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = all
endif
#                -D=$() \

bash.manifest: bash.manifest.template
	gramine-manifest \
                -Dlog_level=$(G_LOG_LEVEL) \
                -Dexecdir=$(shell dirname $(shell which bash)) \
                -Darch_libdir=$(ARCH_LIBDIR) \
                -Dpython_home=$(PYTHON_HOME) \
                -Djdk_home=$(JDK_HOME) \
		-Dg_sgx_size=$(G_SGX_SIZE) \
		-Dg_sgx_thread_num=$(G_SGX_THREAD_NUM) \
		-Dspark_home=$(SPARK_HOME) \
		-Dspark_user=$(SPARK_USER) \
		-Dspark_local_ip=$(SPARK_LOCAL_IP) \
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
