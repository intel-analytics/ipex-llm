/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 */

#define _GNU_SOURCE

#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define USE_STDLIB
#include "api.h"
#include "pf_util.h"
#include "protected_files.h"
#include "protected_files_format.h"
#include "util.h"

/* Tamper with a PF in various ways for testing purposes. The PF is assumed to be valid and have at
 * least enough data to contain two MHT nodes. */

/* Command line options */
struct option g_options[] = {
    { "input", required_argument, 0, 'i' },
    { "output", required_argument, 0, 'o' },
    { "wrap-key", required_argument, 0, 'w' },
    { "verbose", no_argument, 0, 'v' },
    { "help", no_argument, 0, 'h' },
    { 0, 0, 0, 0 }
};

static void usage(void) {
    INFO("\nUsage: pf_tamper [options]\n");
    INFO("\nAvailable options:\n");
    INFO("  --help, -h           Display this help\n");
    INFO("  --verbose, -v        Enable verbose output\n");
    INFO("  --wrap-key, -w PATH  Path to wrap key file\n");
    INFO("  --input, -i PATH     Source file to be tampered with (must be a valid PF)\n");
    INFO("  --output, -o PATH    Directory where modified files will be written to\n");
}

#define FATAL(fmt, ...) do { \
    ERROR(fmt, ##__VA_ARGS__); \
    exit(-1); \
} while (0)

size_t g_input_size = 0;
char* g_input_name = NULL;
void* g_input_data = MAP_FAILED;
char* g_output_dir = NULL;
char* g_output_path = NULL;
size_t g_output_path_size = 0;
pf_key_t g_wrap_key;
pf_key_t g_meta_key;

static pf_iv_t g_empty_iv = {0};

static void derive_main_key(const pf_key_t* kdk, const pf_keyid_t* key_id, pf_key_t* out_key) {
    kdf_input_t buf = {0};
    pf_status_t status;

    buf.index = 1;
    strncpy(buf.label, METADATA_KEY_NAME, MAX_LABEL_SIZE);
    COPY_ARRAY(buf.nonce, *key_id);
    buf.output_len = 0x80;

    status = mbedtls_aes_cmac(kdk, &buf, sizeof(buf), out_key);
    if (PF_FAILURE(status))
        FATAL("key derivation failed\n");
}

static void make_output_path(const char* suffix) {
    snprintf(g_output_path, g_output_path_size, "%s/%s.%s", g_output_dir, g_input_name, suffix);
    INFO("[*] %s\n", g_output_path);
}

/* PF layout (node size is PF_NODE_SIZE):
 * - Node 0: metadata (metadata_node_t)
 *   - metadata_plain_t
 *   - metadata_encrypted_t (may include MD_USER_DATA_SIZE bytes of data)
 *   - metadata_padding_t
 * - Node 1: MHT (mht_node_t)
 * - Node 2-97: data (ATTACHED_DATA_NODES_COUNT == 96)
 * - Node 98: MHT
 * - Node 99-195: data
 * - ...
 */
static void truncate_file(const char* suffix, size_t output_size) {
    int ret;

    make_output_path(suffix);

    if (output_size < g_input_size) {
        ret = write_file(g_output_path, output_size, g_input_data);
    } else {
        ret = write_file(g_output_path, g_input_size, g_input_data);
        if (ret < 0)
            goto out;
        ret = truncate(g_output_path, output_size);
    }
out:
    if (ret < 0)
        FATAL("truncate_file failed: %d\n", ret);
}

#define FIELD_SIZEOF(t, f) (sizeof(((t*)0)->f))
#define FIELD_TRUNCATED(t, f) (offsetof(t, f) + (FIELD_SIZEOF(t, f) / 2))
#define DATA_CRYPTO_SIZE (FIELD_SIZEOF(mht_node_t, data_nodes_crypto))

static void tamper_truncate(void) {
    size_t mdps = sizeof(metadata_plain_t);
    DBG("size(metadata_plain_t)             = 0x%04lx\n", sizeof(metadata_plain_t));
    DBG("metadata_plain_t.file_id           : 0x%04lx (0x%04lx)\n",
        offsetof(metadata_plain_t, file_id), FIELD_SIZEOF(metadata_plain_t, file_id));
    DBG("metadata_plain_t.major_version     : 0x%04lx (0x%04lx)\n",
        offsetof(metadata_plain_t, major_version), FIELD_SIZEOF(metadata_plain_t, major_version));
    DBG("metadata_plain_t.minor_version     : 0x%04lx (0x%04lx)\n",
        offsetof(metadata_plain_t, minor_version), FIELD_SIZEOF(metadata_plain_t, minor_version));
    DBG("metadata_plain_t.metadata_key_id   : 0x%04lx (0x%04lx)\n",
        offsetof(metadata_plain_t, metadata_key_id),
        FIELD_SIZEOF(metadata_plain_t, metadata_key_id));
    DBG("metadata_plain_t.metadata_gmac     : 0x%04lx (0x%04lx)\n",
        offsetof(metadata_plain_t, metadata_gmac),
        FIELD_SIZEOF(metadata_plain_t, metadata_gmac));

    DBG("size(metadata_encrypted_t)         = 0x%04lx\n", sizeof(metadata_encrypted_t));
    DBG("metadata_encrypted_t.path          : 0x%04lx (0x%04lx)\n",
        mdps + offsetof(metadata_encrypted_t, path),
        FIELD_SIZEOF(metadata_encrypted_t, path));
    DBG("metadata_encrypted_t.size          : 0x%04lx (0x%04lx)\n",
        mdps + offsetof(metadata_encrypted_t, size), FIELD_SIZEOF(metadata_encrypted_t, size));
    DBG("metadata_encrypted_t.mht_key       : 0x%04lx (0x%04lx)\n",
        mdps + offsetof(metadata_encrypted_t, mht_key),
        FIELD_SIZEOF(metadata_encrypted_t, mht_key));
    DBG("metadata_encrypted_t.mht_gmac      : 0x%04lx (0x%04lx)\n",
        mdps + offsetof(metadata_encrypted_t, mht_gmac),
        FIELD_SIZEOF(metadata_encrypted_t, mht_gmac));
    DBG("metadata_encrypted_t.data          : 0x%04lx (0x%04lx)\n",
        mdps + offsetof(metadata_encrypted_t, data), FIELD_SIZEOF(metadata_encrypted_t, data));

    DBG("size(metadata_padding_t)           = 0x%04lx\n", sizeof(metadata_padding_t));
    DBG("metadata_padding_t                 : 0x%04lx (0x%04lx)\n",
        mdps + sizeof(metadata_encrypted_t), sizeof(metadata_padding_t));

    /* node 0: metadata + 3k of user data */
    /* plain metadata */
    truncate_file("trunc_meta_plain_0", 0);
    truncate_file("trunc_meta_plain_1", FIELD_TRUNCATED(metadata_plain_t, file_id));
    truncate_file("trunc_meta_plain_2", offsetof(metadata_plain_t, major_version));
    truncate_file("trunc_meta_plain_3", offsetof(metadata_plain_t, minor_version));
    truncate_file("trunc_meta_plain_4", offsetof(metadata_plain_t, metadata_key_id));
    truncate_file("trunc_meta_plain_5", FIELD_TRUNCATED(metadata_plain_t, metadata_key_id));
    truncate_file("trunc_meta_plain_6", offsetof(metadata_plain_t, metadata_gmac));
    truncate_file("trunc_meta_plain_7", FIELD_TRUNCATED(metadata_plain_t, metadata_gmac));

    /* encrypted metadata */
    truncate_file("trunc_meta_enc_0", mdps + offsetof(metadata_encrypted_t, path));
    truncate_file("trunc_meta_enc_1", mdps + FIELD_TRUNCATED(metadata_encrypted_t, path));
    truncate_file("trunc_meta_enc_2", mdps + offsetof(metadata_encrypted_t, size));
    truncate_file("trunc_meta_enc_3", mdps + FIELD_TRUNCATED(metadata_encrypted_t, size));
    truncate_file("trunc_meta_enc_4", mdps + offsetof(metadata_encrypted_t, mht_key));
    truncate_file("trunc_meta_enc_5", mdps + FIELD_TRUNCATED(metadata_encrypted_t, mht_key));
    truncate_file("trunc_meta_enc_6", mdps + offsetof(metadata_encrypted_t, mht_gmac));
    truncate_file("trunc_meta_enc_7", mdps + FIELD_TRUNCATED(metadata_encrypted_t, mht_gmac));
    truncate_file("trunc_meta_enc_8", mdps + offsetof(metadata_encrypted_t, data));
    truncate_file("trunc_meta_enc_9", mdps + FIELD_TRUNCATED(metadata_encrypted_t, data));

    /* padding */
    truncate_file("trunc_meta_pad_0", mdps + sizeof(metadata_encrypted_t));
    truncate_file("trunc_meta_pad_1", mdps + sizeof(metadata_encrypted_t)
                  + sizeof(metadata_padding_t) / 2);

    /* node 1: mht root */
    /* after node 0 */
    truncate_file("trunc_mht_0", PF_NODE_SIZE);
    /* middle of data_nodes_crypto[0].key */
    truncate_file("trunc_mht_1", PF_NODE_SIZE + PF_KEY_SIZE / 2);
    /* after data_nodes_crypto[0].key */
    truncate_file("trunc_mht_2", PF_NODE_SIZE + PF_KEY_SIZE);
    /* middle of data_nodes_crypto[0].gmac */
    truncate_file("trunc_mht_3", PF_NODE_SIZE + PF_KEY_SIZE + PF_MAC_SIZE / 2);
    /* after data_nodes_crypto[0].gmac */
    truncate_file("trunc_mht_4", PF_NODE_SIZE + PF_KEY_SIZE + PF_MAC_SIZE);
    /* after data_nodes_crypto */
    truncate_file("trunc_mht_5", PF_NODE_SIZE + DATA_CRYPTO_SIZE);
    /* middle of mht_nodes_crypto[0].key */
    truncate_file("trunc_mht_6", PF_NODE_SIZE + DATA_CRYPTO_SIZE + PF_KEY_SIZE / 2);
    /* after mht_nodes_crypto[0].key */
    truncate_file("trunc_mht_7", PF_NODE_SIZE + DATA_CRYPTO_SIZE + PF_KEY_SIZE);
    /* middle of mht_nodes_crypto[0].gmac */
    truncate_file("trunc_mht_8", PF_NODE_SIZE + DATA_CRYPTO_SIZE + PF_KEY_SIZE + PF_MAC_SIZE / 2);
    /* after mht_nodes_crypto[0].gmac */
    truncate_file("trunc_mht_9", PF_NODE_SIZE + DATA_CRYPTO_SIZE + PF_KEY_SIZE + PF_MAC_SIZE);

    /* node 2-3: data #0, #1 */
    /* after mht root */
    truncate_file("trunc_data_0", 2 * PF_NODE_SIZE);
    /* middle of data #0 */
    truncate_file("trunc_data_1", 2 * PF_NODE_SIZE + PF_NODE_SIZE / 2);
    /* after data #0 */
    truncate_file("trunc_data_2", 3 * PF_NODE_SIZE);
    /* middle of data #1 */
    truncate_file("trunc_data_3", 3 * PF_NODE_SIZE + PF_NODE_SIZE / 2);

    /* extend */
    truncate_file("extend_0", g_input_size + 1);
    truncate_file("extend_1", g_input_size + PF_NODE_SIZE / 2);
    truncate_file("extend_2", g_input_size + PF_NODE_SIZE);
    truncate_file("extend_3", g_input_size + PF_NODE_SIZE + PF_NODE_SIZE / 2);
}

/* returns mmap'd output contents */
static void* create_output(const char* path) {
    void* mem = MAP_FAILED;
    int fd = open(path, O_RDWR|O_CREAT, 0664);
    if (fd < 0)
        FATAL("Failed to open output file '%s': %s\n", path, strerror(errno));

    if (ftruncate(fd, g_input_size) < 0)
        FATAL("Failed to ftruncate output file '%s': %s\n", path, strerror(errno));

    mem = mmap(NULL, g_input_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED)
        FATAL("Failed to mmap output file '%s': %s\n", path, strerror(errno));

    memcpy(mem, g_input_data, g_input_size);

    close(fd);
    return mem;
}

static void pf_decrypt(const void* encrypted, size_t size, const pf_key_t* key, const pf_mac_t* mac,
                       void* decrypted, const char* msg) {
    pf_status_t status = mbedtls_aes_gcm_decrypt(key, &g_empty_iv, NULL, 0,
                                                 encrypted, size,
                                                 decrypted, mac);
    if (PF_FAILURE(status))
        FATAL("decrypting %s failed\n", msg);
}

static void pf_encrypt(const void* decrypted, size_t size, const pf_key_t* key, pf_mac_t* mac,
                       void* encrypted, const char* msg) {
    pf_status_t status = mbedtls_aes_gcm_encrypt(key, &g_empty_iv, NULL, 0,
                                                 decrypted, size,
                                                 encrypted, mac);
    if (PF_FAILURE(status))
        FATAL("encrypting %s failed\n", msg);
}

/* copy input PF and apply some modifications */
#define __BREAK_PF(suffix, ...) do { \
    make_output_path(suffix); \
    meta = create_output(g_output_path); \
    out = (uint8_t*)meta; \
    pf_decrypt(&meta->encrypted_part, sizeof(meta->encrypted_part), &g_meta_key, \
               &meta->plain_part.metadata_gmac, meta_dec, "metadata"); \
    mht_enc = (mht_node_t*)(out + PF_NODE_SIZE); \
    pf_decrypt(mht_enc, sizeof(*mht_enc), &meta_dec->mht_key, &meta_dec->mht_gmac, mht_dec, \
               "mht"); \
    __VA_ARGS__ \
    munmap(meta, g_input_size); \
} while (0)

/* if update is true, also create a file with correct metadata MAC */
#define BREAK_PF(suffix, update, ...) do { \
    __BREAK_PF(suffix, __VA_ARGS__); \
    if (update) { \
        __BREAK_PF(suffix "_fixed", __VA_ARGS__ { \
                       pf_encrypt(meta_dec, sizeof(*meta_dec), &g_meta_key, \
                                  &meta->plain_part.metadata_gmac, meta->encrypted_part, \
                                  "metadata"); \
                   } ); \
    } \
} while (0)

#define BREAK_MHT(suffix, ...) do { \
    __BREAK_PF(suffix, __VA_ARGS__ { \
                   pf_encrypt(mht_dec, sizeof(*mht_dec), &meta_dec->mht_key, &meta_dec->mht_gmac, \
                              mht_enc, "mht"); \
               } ); \
} while (0)

#define LAST_BYTE(array) (((uint8_t*)&array)[sizeof(array) - 1])

static void tamper_modify(void) {
    metadata_node_t* meta = NULL;
    uint8_t* out = NULL;
    metadata_encrypted_t* meta_dec = malloc(sizeof(*meta_dec));
    if (!meta_dec)
        FATAL("Out of memory\n");
    mht_node_t* mht_enc = NULL;
    mht_node_t* mht_dec = malloc(sizeof(*mht_dec));
    if (!mht_dec)
        FATAL("Out of memory\n");

    /* plain part of the metadata isn't covered by the MAC so no point updating it */
    BREAK_PF("meta_plain_id_0", /*update=*/false,
             { meta->plain_part.file_id = 0; });
    BREAK_PF("meta_plain_id_1", /*update=*/false,
             { meta->plain_part.file_id = UINT64_MAX; });
    BREAK_PF("meta_plain_version_0", /*update=*/false,
             { meta->plain_part.major_version = 0; });
    BREAK_PF("meta_plain_version_1", /*update=*/false,
             { meta->plain_part.major_version = 0xff; });
    BREAK_PF("meta_plain_version_2", /*update=*/false,
             { meta->plain_part.minor_version = 0xff; });

    /* metadata_key_id is the keying material for encrypted metadata key derivation, so create also
     * PFs with updated MACs */
    BREAK_PF("meta_plain_keyid_0", /*update=*/true,
             { meta->plain_part.metadata_key_id[0] ^= 1; });
    BREAK_PF("meta_plain_keyid_1", /*update=*/true,
             { LAST_BYTE(meta->plain_part.metadata_key_id) ^= 0xfe; });
    BREAK_PF("meta_plain_mac_0", /*update=*/true,
             { meta->plain_part.metadata_gmac[0] ^= 0xfe; });
    BREAK_PF("meta_plain_mac_1", /*update=*/true,
             { LAST_BYTE(meta->plain_part.metadata_gmac) &= 1; });

    BREAK_PF("meta_enc_filename_0", /*update=*/true,
             { meta_dec->path[0] = 0; });
    BREAK_PF("meta_enc_filename_1", /*update=*/true,
             { meta_dec->path[0] ^= 1; });
    BREAK_PF("meta_enc_filename_2", /*update=*/true,
             { LAST_BYTE(meta_dec->path) ^= 0xfe; });
    BREAK_PF("meta_enc_size_0", /*update=*/true,
             { meta_dec->size = 0; });
    BREAK_PF("meta_enc_size_1", /*update=*/true,
             { meta_dec->size = g_input_size - 1; });
    BREAK_PF("meta_enc_size_2", /*update=*/true,
             { meta_dec->size = g_input_size + 1; });
    BREAK_PF("meta_enc_size_3", /*update=*/true,
             { meta_dec->size = UINT64_MAX; });
    BREAK_PF("meta_enc_mht_key_0", /*update=*/true,
             { meta_dec->mht_key[0] ^= 1; });
    BREAK_PF("meta_enc_mht_key_1", /*update=*/true,
             { LAST_BYTE(meta_dec->mht_key) ^= 0xfe; });
    BREAK_PF("meta_enc_mht_mac_0", /*update=*/true,
             { meta_dec->mht_gmac[0] ^= 1; });
    BREAK_PF("meta_enc_mht_mac_1", /*update=*/true,
             { LAST_BYTE(meta_dec->mht_gmac) ^= 0xfe; });
    BREAK_PF("meta_enc_data_0", /*update=*/true,
             { meta_dec->data[0] ^= 0xfe; });
    BREAK_PF("meta_enc_data_1", /*update=*/true,
             { LAST_BYTE(meta_dec->data) ^= 1; });

    /* padding is ignored */
    BREAK_PF("meta_padding_0", /*update=*/false,
             { meta->padding[0] ^= 1; });
    BREAK_PF("meta_padding_1", /*update=*/false,
             { LAST_BYTE(meta->padding) ^= 0xfe; });

    BREAK_MHT("mht_0", { mht_dec->data_nodes_crypto[0].key[0] ^= 1; });
    BREAK_MHT("mht_1", { mht_dec->data_nodes_crypto[0].gmac[0] ^= 1; });
    BREAK_MHT("mht_2", { mht_dec->mht_nodes_crypto[0].key[0] ^= 1; });
    BREAK_MHT("mht_3", { mht_dec->mht_nodes_crypto[0].gmac[0] ^= 1; });
    BREAK_MHT("mht_4", { mht_dec->data_nodes_crypto[ATTACHED_DATA_NODES_COUNT - 1].key[0] ^= 1; });
    BREAK_MHT("mht_5", { mht_dec->data_nodes_crypto[ATTACHED_DATA_NODES_COUNT - 1].gmac[0] ^= 1; });
    BREAK_MHT("mht_6", { mht_dec->mht_nodes_crypto[CHILD_MHT_NODES_COUNT - 1].key[0] ^= 1; });
    BREAK_MHT("mht_7", { mht_dec->mht_nodes_crypto[CHILD_MHT_NODES_COUNT - 1].gmac[0] ^= 1; });
    BREAK_MHT("mht_8", {
        gcm_crypto_data_t crypto;
        memcpy(&crypto, &mht_dec->data_nodes_crypto[0], sizeof(crypto));
        memcpy(&mht_dec->data_nodes_crypto[0], &mht_dec->data_nodes_crypto[1], sizeof(crypto));
        memcpy(&mht_dec->data_nodes_crypto[1], &crypto, sizeof(crypto));
    });
    BREAK_MHT("mht_9", {
        gcm_crypto_data_t crypto;
        memcpy(&crypto, &mht_dec->mht_nodes_crypto[0], sizeof(crypto));
        memcpy(&mht_dec->mht_nodes_crypto[0], &mht_dec->mht_nodes_crypto[1], sizeof(crypto));
        memcpy(&mht_dec->mht_nodes_crypto[1], &crypto, sizeof(crypto));
    });

    /* data nodes start from node #2 */
    BREAK_PF("data_0", /*update=*/false,
             { *(out + 2 * PF_NODE_SIZE) ^= 1; });
    BREAK_PF("data_1", /*update=*/false,
             { *(out + 3 * PF_NODE_SIZE - 1) ^= 1; });
    BREAK_PF("data_2", /*update=*/false, {
        /* swap data nodes */
        memcpy(out + 2 * PF_NODE_SIZE, g_input_data + 3 * PF_NODE_SIZE, PF_NODE_SIZE);
        memcpy(out + 3 * PF_NODE_SIZE, g_input_data + 2 * PF_NODE_SIZE, PF_NODE_SIZE);
    });

    free(mht_dec);
    free(meta_dec);
}

int main(int argc, char* argv[]) {
    int ret = -1;

    int option          = 0;
    char* input_path    = NULL;
    char* wrap_key_path = NULL;
    int input_fd        = -1;

    while (true) {
        option = getopt_long(argc, argv, "i:o:w:vh", g_options, NULL);
        if (option == -1)
            break;

        switch (option) {
            case 'i':
                input_path = optarg;
                break;
            case 'o':
                g_output_dir = optarg;
                break;
            case 'w':
                wrap_key_path = optarg;
                break;
            case 'v':
                set_verbose(true);
                break;
            case 'h':
                usage();
                return 0;
            default:
                ERROR("Unknown option: %c\n", option);
                usage();
        }
    }

    if (!input_path) {
        ERROR("Input path not specified\n");
        usage();
        goto out;
    }

    if (!g_output_dir) {
        ERROR("Output path not specified\n");
        usage();
        goto out;
    }

    if (!wrap_key_path) {
        ERROR("Wrap key path not specified\n");
        usage();
        goto out;
    }

    input_fd = open(input_path, O_RDONLY);
    if (input_fd < 0) {
        ERROR("Failed to open input file '%s': %s\n", input_path, strerror(errno));
        goto out;
    }

    ssize_t input_size = get_file_size(input_fd);
    if (input_size < 0) {
        ERROR("Failed to stat input file '%s': %s\n", input_path, strerror(errno));
        goto out;
    }
    g_input_size = input_size;

    g_input_data = mmap(NULL, g_input_size, PROT_READ, MAP_PRIVATE, input_fd, 0);
    if (g_input_data == MAP_FAILED) {
        ERROR("Failed to mmap input file '%s': %s\n", input_path, strerror(errno));
        goto out;
    }

    load_wrap_key(wrap_key_path, &g_wrap_key);
    derive_main_key(&g_wrap_key, &((metadata_plain_t*)g_input_data)->metadata_key_id,
                    &g_meta_key);

    g_input_name = basename(input_path);
    g_output_path_size = strlen(g_input_name) + strlen(g_output_dir) + 256;
    g_output_path = malloc(g_output_path_size);
    if (!g_output_path) {
        ERROR("No memory\n");
        goto out;
    }

    tamper_truncate();
    tamper_modify();
    ret = 0;

out:
    /* skip cleanup as we are in main() */
    return ret;
}
