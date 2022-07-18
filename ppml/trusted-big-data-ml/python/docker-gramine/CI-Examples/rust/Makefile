ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

.PHONY: all
all: build-rust rust-hyper-http-server.manifest
ifeq ($(SGX),1)
all: rust-hyper-http-server.manifest.sgx rust-hyper-http-server.sig rust-hyper-http-server.token
endif

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif

# We don't attempt to instruct Make on when the executable is stale and needs rebuilding.
# Instead, we make use of Cargo's own incremental compilation, liberally executing Cargo
# itself, with the understanding that it won't actually compile anything in most cases.
#
# One could also integrate Make with the dependency (.d) files Cargo generates within target/
# to only run Cargo when strictly necessary. We have decided that, in this case, the additional
# complexity is not worth it.
#
# Also note that we're compiling in release mode regardless of the DEBUG setting passed
# to Make, as compiling in debug mode results in an order of magnitude's difference in
# performance that makes testing by running a benchmark with ab painful. The primary goal
# of the DEBUG setting is to control Gramine's loglevel.
.PHONY: build-rust
build-rust:
	cargo build --release

SELF_EXE = target/release/rust-hyper-http-server

rust-hyper-http-server.manifest: rust-hyper-http-server.manifest.template
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dself_exe=$(SELF_EXE) \
		$< $@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# see the helloworld example for details on this workaround.
rust-hyper-http-server.manifest.sgx rust-hyper-http-server.sig: sgx_sign
	@:

.INTERMEDIATE: sgx_sign
sgx_sign: rust-hyper-http-server.manifest build-rust
	gramine-sgx-sign \
		--manifest $< \
		--output $<.sgx

rust-hyper-http-server.token: rust-hyper-http-server.sig
	gramine-sgx-get-token \
		--output $@ --sig $<

ifeq ($(SGX),)
GRAMINE = gramine-direct
else
GRAMINE = gramine-sgx
endif

.PHONY: start-gramine-server
start-gramine-server: all
	$(GRAMINE) rust-hyper-http-server

.PHONY: clean
clean:
	$(RM) -rf *.token *.sig *.manifest.sgx *.manifest result-* OUTPUT

.PHONY: distclean
distclean: clean
	$(RM) -rf target/ Cargo.lock
