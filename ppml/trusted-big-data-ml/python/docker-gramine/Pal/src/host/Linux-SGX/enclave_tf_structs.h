/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation */

#ifndef ENCLAVE_TF_STRUCTS_H_
#define ENCLAVE_TF_STRUCTS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "list.h"

enum {
    FILE_CHECK_POLICY_STRICT = 0,
    FILE_CHECK_POLICY_ALLOW_ALL_BUT_LOG,
};

typedef struct {
    uint8_t bytes[32];
} sgx_file_hash_t;

typedef struct {
    uint8_t bytes[16];
} sgx_chunk_hash_t;

/*
 * Perhaps confusingly, `struct trusted_file` describes not only "sgx.trusted_files" but also
 * "sgx.allowed_files". For allowed files, `allowed = true`, `chunk_hashes = NULL`, and `uri` can be
 * not only a file but also a directory. TODO: Perhaps split "allowed_files" into a separate struct?
 */
DEFINE_LIST(trusted_file);
struct trusted_file {
    LIST_TYPE(trusted_file) list;
    uint64_t size;
    bool allowed;
    sgx_file_hash_t file_hash;      /* hash over the whole file, retrieved from the manifest */
    sgx_chunk_hash_t* chunk_hashes; /* array of hashes over separate file chunks */
    size_t uri_len;
    char uri[]; /* must be NULL-terminated */
};

#endif /* ENCLAVE_TF_STRUCTS_H_ */
