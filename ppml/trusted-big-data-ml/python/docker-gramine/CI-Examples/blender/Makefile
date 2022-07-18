# assumes this makefile lies in cwd
PWD := $(shell pwd)

ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

BLENDER_DIR = $(PWD)/blender_dir
BLENDER_SRC ?= blender-2.82-linux64.tar.xz
BLENDER_MIRRORS ?= \
    https://ftp.nluug.nl/pub/graphics/blender/release/Blender2.82 \
    https://packages.gramineproject.io/distfiles
BLENDER_SHA256 ?= b13600fa2ca23ea1bba511e3a6599b6792acde80b180707c3ea75db592a9b916
BLENDER_VER = 2.82

DATA_DIR = $(PWD)/data
RUN_DIR = $(PWD)/run_dir

UBUNTU_VER = $(shell lsb_release --short --id)$(shell lsb_release --short --release)

ifeq ($(DEBUG),1)
GRAMINE_LOG_LEVEL = debug
else
GRAMINE_LOG_LEVEL = error
endif


.PHONY: all
all: $(BLENDER_DIR)/blender blender.manifest | $(DATA_DIR)/images
ifeq ($(SGX),1)
all: blender.manifest.sgx blender.sig blender.token
endif

$(BLENDER_DIR)/blender:
	../common_tools/download --output blender.tar.xz --sha256 $(BLENDER_SHA256) \
		$(foreach mirror,$(BLENDER_MIRRORS),--url $(mirror)/$(BLENDER_SRC))
	mkdir $(BLENDER_DIR)
	tar -C $(BLENDER_DIR) --strip-components=1 -xf blender.tar.xz

$(RUN_DIR):
	mkdir -p $@

blender.manifest: blender.manifest.template | $(RUN_DIR)
	gramine-manifest \
		-Dlog_level=$(GRAMINE_LOG_LEVEL) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Ddata_dir=$(DATA_DIR) \
		-Dblender_dir=$(BLENDER_DIR) \
		-Dblender_ver=$(BLENDER_VER) \
		$< >$@

# Make on Ubuntu <= 20.04 doesn't support "Rules with Grouped Targets" (`&:`),
# see the helloworld example for details on this workaround.
blender.sig blender.manifest.sgx: sgx_outputs
	@:

.INTERMEDIATE: sgx_outputs
sgx_outputs: $(BLENDER_DIR)/blender blender.manifest | $(RUN_DIR)
	gramine-sgx-sign \
		--output blender.manifest.sgx \
		--manifest blender.manifest

blender.token: blender.sig
	gramine-sgx-get-token --output $@ --sig $<

$(DATA_DIR)/images:
	mkdir -p $@

.PHONY: check
check: all
	cd $(RUN_DIR) && DATA_DIR=$(DATA_DIR) sh $(PWD)/test_all_scenes.sh

.PHONY: clean
clean:
	$(RM) -r \
		$(RUN_DIR) $(DATA_DIR)/images blender.manifest blender.manifest.sgx blender.sig \
		blender.token

.PHONY: distclean
distclean: clean
	$(RM) -r $(BLENDER_DIR) blender.tar.xz

