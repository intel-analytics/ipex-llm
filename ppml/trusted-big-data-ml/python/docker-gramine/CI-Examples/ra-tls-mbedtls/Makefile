GRAMINEDIR ?= ../..
GRAMINE_PKGLIBDIR ?= /usr/lib/x86_64-linux-gnu/gramine # this is debian/ubuntu specific

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

# for EPID attestation, specify your SPID and linkable/unlinkable attestation policy;
# for DCAP/ECDSA attestation, specify SPID as empty string (linkable value is ignored)
RA_CLIENT_SPID ?=
RA_CLIENT_LINKABLE ?= 0

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
CFLAGS += -O0 -ggdb3
else
GRAMINE_LOG_LEVEL = error
CFLAGS += -O2
endif

CFLAGS += -fPIE
LDFLAGS += -pie

.PHONY: all
all: app epid  # by default, only build EPID because it doesn't rely on additional (DCAP) libs

.PHONY: app
app: ssl/server.crt mbedtls/.mbedtls_configured server.manifest.sgx server.sig server.token client

.PHONY: epid
epid: client_epid.manifest.sgx client_epid.sig client_epid.token

.PHONY: dcap
dcap: client_dcap.manifest.sgx client_dcap.sig client_dcap.token

############################# SSL DATA DEPENDENCY #############################

# SSL data: key and x.509 self-signed certificate
ssl/server.crt: ssl/ca_config.conf
	openssl genrsa -out ssl/ca.key 2048
	openssl req -x509 -new -nodes -key ssl/ca.key -sha256 -days 1024 -out ssl/ca.crt -config ssl/ca_config.conf
	openssl genrsa -out ssl/server.key 2048
	openssl req -new -key ssl/server.key -out ssl/server.csr -config ssl/ca_config.conf
	openssl x509 -req -days 360 -in ssl/server.csr -CA ssl/ca.crt -CAkey ssl/ca.key -CAcreateserial -out ssl/server.crt

############################# MBEDTLS DEPENDENCY ##############################

#
# This download is done to get headers in include/, because we currently fail
# to provide the headers.
# TODO: install headers, and `make install` before copying them
#

MBEDTLS_VERSION ?= 2.26.0
MBEDTLS_SRC ?= mbedtls-$(MBEDTLS_VERSION).tar.gz
MBEDTLS_URI ?= \
	https://github.com/ARMmbed/mbedtls/archive \
	https://packages.gramineproject.io/distfiles
MBEDTLS_HASH ?= 35d8d87509cd0d002bddbd5508b9d2b931c5e83747d087234cc7ad551d53fe05

ifeq ($(DEBUG),1)
MBED_BUILD_TYPE=Debug
else
MBED_BUILD_TYPE=Release
endif

$(MBEDTLS_SRC):
	../common_tools/download --output $@ $(foreach mirror,$(MBEDTLS_URI),--url $(mirror)/$(MBEDTLS_SRC)) --sha256 $(MBEDTLS_HASH)

.SECONDARY: mbedtls/.mbedtls_downloaded
mbedtls/.mbedtls_downloaded: $(MBEDTLS_SRC)
	tar --touch -xzf $(MBEDTLS_SRC)
	mv mbedtls-mbedtls-$(MBEDTLS_VERSION) mbedtls
	touch $@

.SECONDARY: mbedtls/.mbedtls_configured
mbedtls/.mbedtls_configured: mbedtls/.mbedtls_downloaded
	cd mbedtls && ./scripts/config.pl set MBEDTLS_CMAC_C MBEDTLS_ERROR_C
	touch $@

######################### CLIENT/SERVER EXECUTABLES ###########################

CFLAGS += -I./mbedtls/include $(shell pkg-config --cflags mbedtls_gramine)
LDFLAGS += -ldl $(shell pkg-config --libs mbedtls_gramine)

server: src/server.c mbedtls/.mbedtls_configured
	$(CC) $< $(CFLAGS) $(LDFLAGS) -o $@

client: src/client.c mbedtls/.mbedtls_configured
	$(CC) $< $(CFLAGS) $(LDFLAGS) -o $@

############################### SERVER MANIFEST ###############################

server.manifest: server.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dra_client_spid=$(RA_CLIENT_SPID) \
		-Dra_client_linkable=$(RA_CLIENT_LINKABLE) \
		$< > $@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# see the helloworld example for details on this workaround.
server.manifest.sgx server.sig: sgx_sign_server
	@:

.INTERMEDIATE: sgx_sign_server
sgx_sign_server: server.manifest server
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

server.token: server.sig
	gramine-sgx-get-token --output $@ --sig $<

########################### CLIENT (DCAP) MANIFEST ############################

client_dcap.manifest: client.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		$< >$@

client_dcap.manifest.sgx client_dcap.sig: sgx_sign_client_dcap
	@:

.INTERMEDIATE: sgx_sign_client_dcap
sgx_sign_client_dcap: client_dcap.manifest client
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

client_dcap.token: client_dcap.sig
	gramine-sgx-get-token --output $@ --sig $<

########################### CLIENT (EPID) MANIFEST ############################

client_epid.manifest: client.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		$< >$@

client_epid.manifest.sgx client_epid.sig: sgx_sign_client_epid
	@:

.INTERMEDIATE: sgx_sign_client_epid
sgx_sign_client_epid: client_epid.manifest client
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

client_epid.token: client_epid.sig
	gramine-sgx-get-token --output $@ --sig $<

############################### SGX CHECKS FOR CI #############################

.PHONY: check_epid
check_epid: app epid
	gramine-sgx server epid & SERVER_ID=$$!; \
	sleep 30; \
	./client epid > OUTPUT; \
	./client epid 0 0 0 0 >> OUTPUT; \
	kill -9 $$SERVER_ID
	@grep -q "using default SGX-measurement verification callback" OUTPUT && echo "[ Success 1/4 ]"
	@grep -q "using our own SGX-measurement verification callback" OUTPUT && echo "[ Success 2/4 ]"
	@grep -q "Verifying peer X.509 certificate... ok" OUTPUT && echo "[ Success 3/4 ]"
	@(exit `grep -c "failed" "OUTPUT"`) && echo "[ Success 4/4 ]"
	@rm OUTPUT

.PHONY: check_epid_fail
check_epid_fail: app epid
	gramine-sgx server epid dummy-option >/dev/null & SERVER_ID=$$!; \
	sleep 30; \
	./client epid && exit 1 || echo "[ Success 1/1 ]"; \
	kill -9 $$SERVER_ID

.PHONY: check_dcap
check_dcap: app dcap
	gramine-sgx server dcap >/dev/null & SERVER_ID=$$!; \
	sleep 30; \
	./client dcap > OUTPUT; \
	./client dcap 0 0 0 0 >> OUTPUT; \
	kill -9 $$SERVER_ID
	@grep -q "using default SGX-measurement verification callback" OUTPUT && echo "[ Success 1/4 ]"
	@grep -q "using our own SGX-measurement verification callback" OUTPUT && echo "[ Success 2/4 ]"
	@grep -q "Verifying peer X.509 certificate... ok" OUTPUT && echo "[ Success 3/4 ]"
	@(exit `grep -c "failed" "OUTPUT"`) && echo "[ Success 4/4 ]"
	@rm OUTPUT

.PHONY: check_dcap_fail
check_dcap_fail: app dcap
	gramine-sgx server dcap dummy-option >/dev/null & SERVER_ID=$$!; \
	sleep 30; \
	./client dcap && exit 1 || echo "[ Success 1/1 ]"; \
	kill -9 $$SERVER_ID

################################## CLEANUP ####################################

.PHONY: clean
clean:
	$(RM) -r \
		*.token *.sig *.manifest.sgx *.manifest server client *.so *.so.* OUTPUT

.PHONY: distclean
distclean: clean
	$(RM) -r mbedtls/ *.tar.gz ssl/ca.* ssl/server.*
