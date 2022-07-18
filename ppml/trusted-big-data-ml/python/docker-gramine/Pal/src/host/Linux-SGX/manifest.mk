SGX_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# sgx manifest.sgx/sig/token
drop_manifest_suffix = $(filter-out manifest,$(sort $(patsubst %.manifest,%,$(1))))
expand_target_to_token = $(addsuffix .token,$(call drop_manifest_suffix,$(1)))
expand_target_to_sig = $(addsuffix .sig,$(call drop_manifest_suffix,$(1)))
expand_target_to_sgx = $(addsuffix .manifest.sgx,$(call drop_manifest_suffix,$(1)))

%.token: %.sig
	$(call cmd,sgx_get_token)

%.sig %.manifest.sgx: %.manifest %.manifest.sgx.d
	$(call cmd,sgx_sign)

.PRECIOUS: %.manifest.sgx.d
%.manifest.sgx.d: %.manifest
	$(call cmd,manifest_gen_depend)

ifeq ($(filter %clean,$(MAKECMDGOALS)),)
ifeq ($(target),)
$(error define "target" variable for manifest.sgx dependency calculation)
endif
include $(addsuffix .manifest.sgx.d,$(call drop_manifest_suffix,$(target)))
endif
