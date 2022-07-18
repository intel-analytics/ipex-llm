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

.PHONY: clients
clients: secret_prov_min_client secret_prov_client secret_prov_pf_client

.PHONY: all
all: app epid  # by default, only build EPID because it doesn't rely on additional (DCAP) libs

.PHONY: app
app: \
	 ssl/server.crt \
     secret_prov_min_client.manifest.sgx secret_prov_min_client.sig secret_prov_min_client.token \
     secret_prov_client.manifest.sgx     secret_prov_client.sig     secret_prov_client.token     \
     secret_prov_pf_client.manifest.sgx  secret_prov_pf_client.sig  secret_prov_pf_client.token

.PHONY: epid
epid: secret_prov_server_epid

.PHONY: dcap
dcap: secret_prov_server_dcap

############################# SSL DATA DEPENDENCY #############################

# SSL data: key and x.509 self-signed certificate
ssl/server.crt: ssl/ca_config.conf
	openssl genrsa -out ssl/ca.key 2048
	openssl req -x509 -new -nodes -key ssl/ca.key -sha256 -days 1024 -out ssl/ca.crt -config ssl/ca_config.conf
	openssl genrsa -out ssl/server.key 2048
	openssl req -new -key ssl/server.key -out ssl/server.csr -config ssl/ca_config.conf
	openssl x509 -req -days 360 -in ssl/server.csr -CA ssl/ca.crt -CAkey ssl/ca.key -CAcreateserial -out ssl/server.crt

######################### CLIENT/SERVER EXECUTABLES ###########################

CFLAGS += -Wall -std=c11 -I$(GRAMINEDIR)/Pal/src/host/Linux-SGX/tools/ra-tls \
          $(shell pkg-config --cflags mbedtls_gramine)
LDFLAGS += $(shell pkg-config --libs mbedtls_gramine)

secret_prov_server_epid: src/secret_prov_server.c
	$(CC) $< $(CFLAGS) $(LDFLAGS) -lsecret_prov_verify_epid -pthread -o $@

# linker option --no-as-needed is required because SGX DCAP library (libsgx_dcap_quoteverify.so)
# does dlopen() instead of directly linking against libsgx_urts.so, and without this option
# compilers remove the "seemingly unused" libsgx_urts.so
secret_prov_server_dcap: src/secret_prov_server.c
	$(CC) $< $(CFLAGS) $(LDFLAGS) -Wl,--no-as-needed -lsgx_urts -lsecret_prov_verify_dcap -pthread -o $@

secret_prov_client: src/secret_prov_client.c
	$(CC) $< $(CFLAGS) $(LDFLAGS) -lsecret_prov_attest -o $@

secret_prov_min_client: src/secret_prov_min_client.c
	$(CC) $< $(CFLAGS) $(LDFLAGS) -o $@

secret_prov_pf_client: src/secret_prov_pf_client.c
	$(CC) $< $(CFLAGS) $(LDFLAGS) -o $@

############################### CLIENT MANIFEST ###############################

secret_prov_client.manifest: secret_prov_client.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dra_client_spid=$(RA_CLIENT_SPID) \
		-Dra_client_linkable=$(RA_CLIENT_LINKABLE) \
		$< > $@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# see the helloworld example for details on this workaround.
secret_prov_client.manifest.sgx secret_prov_client.sig: sgx_sign_secret_prov_client
	@:

.INTERMEDIATE: sgx_sign_secret_prov_client
sgx_sign_secret_prov_client: secret_prov_client.manifest secret_prov_client
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

secret_prov_client.token: secret_prov_client.sig
	gramine-sgx-get-token --output $@ --sig $<

############################# MIN CLIENT MANIFEST #############################

secret_prov_min_client.manifest: secret_prov_min_client.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dra_client_spid=$(RA_CLIENT_SPID) \
		-Dra_client_linkable=$(RA_CLIENT_LINKABLE) \
		$< > $@

secret_prov_min_client.manifest.sgx secret_prov_min_client.sig: sgx_sign_secret_prov_min_client
	@:

.INTERMEDIATE: sgx_sign_secret_prov_min_client
sgx_sign_secret_prov_min_client: secret_prov_min_client.manifest secret_prov_min_client
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

secret_prov_min_client.token: secret_prov_min_client.sig
	gramine-sgx-get-token --output $@ --sig $<

########################## PREPARE PROTECTED FILES ############################

files/input.txt: files/wrap-key files/plain.txt
	gramine-sgx-pf-crypt encrypt -w files/wrap-key -i files/plain.txt -o $@

############################## PF CLIENT MANIFEST #############################

secret_prov_pf_client.manifest: secret_prov_pf_client.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dra_client_spid=$(RA_CLIENT_SPID) \
		-Dra_client_linkable=$(RA_CLIENT_LINKABLE) \
		$< > $@

secret_prov_pf_client.manifest.sgx secret_prov_pf_client.sig: sgx_sign_secret_prov_pf_client
	@:

.INTERMEDIATE: sgx_sign_secret_prov_pf_client
sgx_sign_secret_prov_pf_client: secret_prov_pf_client.manifest secret_prov_pf_client
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

secret_prov_pf_client.token: secret_prov_pf_client.sig
	gramine-sgx-get-token --output $@ --sig $<

############################# SGX CHECKS FOR CI ###############################

.PHONY: check_epid
check_epid: app epid files/input.txt
	./secret_prov_server_epid >/dev/null & SERVER_ID=$$!; \
	sleep 3; \
	gramine-sgx ./secret_prov_min_client > OUTPUT; \
	gramine-sgx ./secret_prov_client >> OUTPUT; \
	gramine-sgx ./secret_prov_pf_client >> OUTPUT; \
	kill -9 $$SERVER_ID;
	@grep "Received secret = 'ffeeddccbbaa99887766554433221100'" OUTPUT && echo "[ Success 1/4 ]"
	@grep "Received secret1 = 'ffeeddccbbaa99887766554433221100', secret2 = '42'" OUTPUT && echo "[ Success 2/4 ]"
	@grep "\[parent\] Read from protected file: 'helloworld'" OUTPUT && echo "[ Success 3/4 ]"
	@grep "\[child\] Read from protected file: 'helloworld'" OUTPUT && echo "[ Success 4/4 ]"
	@rm OUTPUT

.PHONY: check_dcap
check_dcap: app dcap files/input.txt
	./secret_prov_server_dcap >/dev/null & SERVER_ID=$$!; \
	sleep 3; \
	gramine-sgx ./secret_prov_min_client > OUTPUT; \
	gramine-sgx ./secret_prov_client >> OUTPUT; \
	gramine-sgx ./secret_prov_pf_client >> OUTPUT; \
	kill -9 $$SERVER_ID;
	@grep "Received secret = 'ffeeddccbbaa99887766554433221100'" OUTPUT && echo "[ Success 1/4 ]"
	@grep "Received secret1 = 'ffeeddccbbaa99887766554433221100', secret2 = '42'" OUTPUT && echo "[ Success 2/4 ]"
	@grep "\[parent\] Read from protected file: 'helloworld'" OUTPUT && echo "[ Success 3/4 ]"
	@grep "\[child\] Read from protected file: 'helloworld'" OUTPUT && echo "[ Success 4/4 ]"
	@rm OUTPUT

################################## CLEANUP ####################################

.PHONY: clean
clean:
	$(RM) -r \
		*.token *.sig *.manifest.sgx *.manifest *.so *.so.* OUTPUT \
		secret_prov_server_epid secret_prov_server_dcap \
		secret_prov_client secret_prov_min_client secret_prov_pf_client

.PHONY: distclean
distclean: clean
	$(RM) -r *.tar.gz files/input.txt ssl/ca.* ssl/server.*
