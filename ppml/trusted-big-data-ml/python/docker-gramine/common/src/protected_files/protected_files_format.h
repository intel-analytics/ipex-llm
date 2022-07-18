/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 * Copyright (C) 2020 Intel Corporation
 */

#ifndef PROTECTED_FILES_FORMAT_H_
#define PROTECTED_FILES_FORMAT_H_

#ifdef USE_STDLIB
#include <assert.h>
#else
#include "assert.h"
#endif

#include <limits.h>

#include "list.h"
#include "protected_files.h"

#define PF_FILE_ID       0x46505f5346415247 /* GRAFS_PF */
#define PF_MAJOR_VERSION 0x01
#define PF_MINOR_VERSION 0x00

#define METADATA_KEY_NAME "SGX-PROTECTED-FS-METADATA-KEY"
#define MAX_LABEL_SIZE    64

static_assert(sizeof(METADATA_KEY_NAME) <= MAX_LABEL_SIZE, "label too long");

#pragma pack(push, 1)

typedef struct _metadata_plain {
    uint64_t   file_id;
    uint8_t    major_version;
    uint8_t    minor_version;
    pf_keyid_t metadata_key_id;
    pf_mac_t   metadata_gmac; /* GCM mac */
} metadata_plain_t;

#define PATH_MAX_SIZE (260 + 512)

// these are all defined as relative to node size, so we can decrease node size in tests
// and have deeper tree
#define MD_USER_DATA_SIZE (PF_NODE_SIZE * 3 / 4) // 3072
static_assert(MD_USER_DATA_SIZE == 3072, "bad struct size");

typedef struct _metadata_encrypted {
    char     path[PATH_MAX_SIZE];
    uint64_t size;
    pf_key_t mht_key;
    pf_mac_t mht_gmac;
    uint8_t  data[MD_USER_DATA_SIZE];
} metadata_encrypted_t;

typedef uint8_t metadata_encrypted_blob_t[sizeof(metadata_encrypted_t)];

#define METADATA_NODE_SIZE PF_NODE_SIZE

typedef uint8_t metadata_padding_t[METADATA_NODE_SIZE -
                                   (sizeof(metadata_plain_t) + sizeof(metadata_encrypted_blob_t))];

typedef struct _metadata_node {
    metadata_plain_t          plain_part;
    metadata_encrypted_blob_t encrypted_part;
    metadata_padding_t        padding;
} metadata_node_t;

static_assert(sizeof(metadata_node_t) == PF_NODE_SIZE, "sizeof(metadata_node_t)");

typedef struct _data_node_crypto {
    pf_key_t key;
    pf_mac_t gmac;
} gcm_crypto_data_t;

// for PF_NODE_SIZE == 4096, we have 96 attached data nodes and 32 mht child nodes
// for PF_NODE_SIZE == 2048, we have 48 attached data nodes and 16 mht child nodes
// for PF_NODE_SIZE == 1024, we have 24 attached data nodes and 8 mht child nodes
// 3/4 of the node size is dedicated to data nodes
#define ATTACHED_DATA_NODES_COUNT ((PF_NODE_SIZE / sizeof(gcm_crypto_data_t)) * 3 / 4)
static_assert(ATTACHED_DATA_NODES_COUNT == 96, "ATTACHED_DATA_NODES_COUNT");
// 1/4 of the node size is dedicated to child mht nodes
#define CHILD_MHT_NODES_COUNT ((PF_NODE_SIZE / sizeof(gcm_crypto_data_t)) * 1 / 4)
static_assert(CHILD_MHT_NODES_COUNT == 32, "CHILD_MHT_NODES_COUNT");

typedef struct _mht_node {
    gcm_crypto_data_t data_nodes_crypto[ATTACHED_DATA_NODES_COUNT];
    gcm_crypto_data_t mht_nodes_crypto[CHILD_MHT_NODES_COUNT];
} mht_node_t;

static_assert(sizeof(mht_node_t) == PF_NODE_SIZE, "sizeof(mht_node_t)");

typedef struct _data_node {
    uint8_t data[PF_NODE_SIZE];
} data_node_t;

static_assert(sizeof(data_node_t) == PF_NODE_SIZE, "sizeof(data_node_t)");

typedef struct _encrypted_node {
    uint8_t cipher[PF_NODE_SIZE];
} encrypted_node_t;

static_assert(sizeof(encrypted_node_t) == PF_NODE_SIZE, "sizeof(encrypted_node_t)");

#define MAX_PAGES_IN_CACHE 48

typedef enum {
    FILE_MHT_NODE_TYPE  = 1,
    FILE_DATA_NODE_TYPE = 2,
} mht_node_type_e;

// make sure these are the same size
static_assert(sizeof(mht_node_t) == sizeof(data_node_t),
              "sizeof(mht_node_t) == sizeof(data_node_t)");

DEFINE_LIST(_file_node);
typedef struct _file_node {
    LIST_TYPE(_file_node) list;
    uint8_t type;
    uint64_t node_number;
    struct _file_node* parent;
    bool need_writing;
    bool new_node;
    struct {
        uint64_t physical_node_number;
        encrypted_node_t encrypted; // the actual data from the disk
    };
    union { // decrypted data
        mht_node_t mht;
        data_node_t data;
    } decrypted;
} file_node_t;
DEFINE_LISTP(_file_node);

typedef struct {
    uint32_t index;
    char label[MAX_LABEL_SIZE]; // must be NULL terminated
    pf_keyid_t nonce;
    uint32_t output_len; // in bits
} kdf_input_t;

#pragma pack(pop)

#endif /* PROTECTED_FILES_FORMAT_H_ */

