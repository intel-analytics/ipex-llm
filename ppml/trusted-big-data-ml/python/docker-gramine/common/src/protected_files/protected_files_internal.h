/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 * Copyright (C) 2019 Intel Corporation
 */

#ifndef PROTECTED_FILES_INTERNAL_H_
#define PROTECTED_FILES_INTERNAL_H_

#include <limits.h>

#include "list.h"
#include "lru_cache.h"
#include "protected_files.h"
#include "protected_files_format.h"

struct pf_context {
    metadata_node_t file_metadata; // actual data from disk's meta data node
    pf_status_t last_error;
    metadata_encrypted_t encrypted_part_plain; // encrypted part of metadata node, decrypted
    file_node_t root_mht; // the root of the mht is always needed (for files bigger than 3KB)
    pf_handle_t file;
    pf_file_mode_t mode;
    uint64_t offset; // current file position (user's view)
    bool end_of_file;
    uint64_t real_file_size;
    bool need_writing;
    pf_status_t file_status;
    pf_key_t user_kdk_key;
    pf_key_t cur_key;
    lruc_context_t* cache;
#ifdef DEBUG
    char* debug_buffer; // buffer for debug output
#endif
};

/* ipf prefix means "Intel protected files", these are functions from the SGX SDK implementation */
static bool ipf_init_fields(pf_context_t* pf);
static bool ipf_init_existing_file(pf_context_t* pf, const char* path);
static bool ipf_init_new_file(pf_context_t* pf, const char* path);

static bool ipf_read_node(pf_context_t* pf, pf_handle_t handle, uint64_t node_number, void* buffer,
                          uint32_t node_size);
static bool ipf_write_node(pf_context_t* pf, pf_handle_t handle, uint64_t node_number, void* buffer,
                           uint32_t node_size);

static bool ipf_import_metadata_key(pf_context_t* pf, bool restore, pf_key_t* output);
static bool ipf_generate_random_key(pf_context_t* pf, pf_key_t* output);
static bool ipf_restore_current_metadata_key(pf_context_t* pf, pf_key_t* output);

static file_node_t* ipf_get_data_node(pf_context_t* pf);
static file_node_t* ipf_read_data_node(pf_context_t* pf);
static file_node_t* ipf_append_data_node(pf_context_t* pf);
static file_node_t* ipf_get_mht_node(pf_context_t* pf);
static file_node_t* ipf_read_mht_node(pf_context_t* pf, uint64_t mht_node_number);
static file_node_t* ipf_append_mht_node(pf_context_t* pf, uint64_t mht_node_number);

static bool ipf_update_all_data_and_mht_nodes(pf_context_t* pf);
static bool ipf_update_metadata_node(pf_context_t* pf);
static bool ipf_write_all_changes_to_disk(pf_context_t* pf);
static bool ipf_internal_flush(pf_context_t* pf);

static pf_context_t* ipf_open(const char* path, pf_file_mode_t mode, bool create, pf_handle_t file,
                              size_t real_size, const pf_key_t* kdk_key, pf_status_t* status);
static bool ipf_close(pf_context_t* pf);
static size_t ipf_read(pf_context_t* pf, void* ptr, size_t size);
static size_t ipf_write(pf_context_t* pf, const void* ptr, size_t size);
static bool ipf_seek(pf_context_t* pf, uint64_t new_offset);
static void ipf_try_clear_error(pf_context_t* pf);

#endif /* PROTECTED_FILES_INTERNAL_H_ */
