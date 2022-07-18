/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 */

#define _GNU_SOURCE
#define _LARGEFILE64_SOURCE

#include "pf_util.h"

#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <mbedtls/cmac.h>
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/entropy.h>
#include <mbedtls/gcm.h>

#define USE_STDLIB
#include "api.h"
#include "perm.h"
#include "util.h"

/* High-level protected files helper functions. */

/* PF callbacks usable in a standard Linux environment.
   Assume that pf handle is a pointer to file's fd. */

static pf_status_t linux_read(pf_handle_t handle, void* buffer, uint64_t offset, size_t size) {
    int fd = *(int*)handle;
    DBG("linux_read: fd %d, buf %p, offset %zu, size %zu\n", fd, buffer, offset, size);
    if (lseek64(fd, offset, SEEK_SET) < 0) {
        ERROR("lseek64 failed: %s\n", strerror(errno));
        return PF_STATUS_CALLBACK_FAILED;
    }

    size_t buffer_offset = 0;
    while (size > 0) {
        ssize_t ret = read(fd, buffer + buffer_offset, size);
        if (ret == -EINTR)
            continue;

        if (ret < 0) {
            ERROR("read failed: %s\n", strerror(errno));
            return PF_STATUS_CALLBACK_FAILED;
        }

        /* EOF is an error condition, we want to read exactly `size` bytes */
        if (ret == 0) {
            ERROR("EOF\n");
            return PF_STATUS_CALLBACK_FAILED;
        }

        size -= ret;
        buffer_offset += ret;
    }

    return PF_STATUS_SUCCESS;
}

static pf_status_t linux_write(pf_handle_t handle, const void* buffer, uint64_t offset,
                               size_t size) {
    int fd = *(int*)handle;
    DBG("linux_write: fd %d, buf %p, offset %zu, size %zu\n", fd, buffer, offset, size);
    if (lseek64(fd, offset, SEEK_SET) < 0) {
        ERROR("lseek64 failed: %s\n", strerror(errno));
        return PF_STATUS_CALLBACK_FAILED;
    }

    size_t buffer_offset = 0;
    while (size > 0) {
        ssize_t ret = write(fd, buffer + buffer_offset, size);
        if (ret == -EINTR)
            continue;

        if (ret < 0) {
            ERROR("write failed: %s\n", strerror(errno));
            return PF_STATUS_CALLBACK_FAILED;
        }

        /* EOF is an error condition, we want to write exactly `size` bytes */
        if (ret == 0) {
            ERROR("EOF\n");
            return PF_STATUS_CALLBACK_FAILED;
        }

        size -= ret;
        buffer_offset += ret;
    }
    return PF_STATUS_SUCCESS;
}

static pf_status_t linux_truncate(pf_handle_t handle, uint64_t size) {
    int fd = *(int*)handle;
    DBG("linux_truncate: fd %d, size %zu\n", fd, size);
    int ret = ftruncate64(fd, size);
    if (ret < 0) {
        ERROR("ftruncate64 failed: %s\n", strerror(errno));
        return PF_STATUS_CALLBACK_FAILED;
    }

    return PF_STATUS_SUCCESS;
}

/* Crypto callbacks for mbedTLS */

pf_status_t mbedtls_aes_cmac(const pf_key_t* key, const void* input, size_t input_size,
                             pf_mac_t* mac) {
    const mbedtls_cipher_info_t* cipher_info =
        mbedtls_cipher_info_from_type(MBEDTLS_CIPHER_AES_128_ECB);

    int ret = mbedtls_cipher_cmac(cipher_info, (const unsigned char*)key, PF_KEY_SIZE * 8, input,
                                  input_size, (unsigned char*)mac);
    if (ret != 0) {
        ERROR("mbedtls_cipher_cmac failed: %d\n", ret);
        return PF_STATUS_CALLBACK_FAILED;
    }

    return PF_STATUS_SUCCESS;
}

pf_status_t mbedtls_aes_gcm_encrypt(const pf_key_t* key, const pf_iv_t* iv, const void* aad,
                                    size_t aad_size, const void* input, size_t input_size,
                                    void* output, pf_mac_t* mac) {
    pf_status_t status = PF_STATUS_CALLBACK_FAILED;

    mbedtls_gcm_context gcm;
    mbedtls_gcm_init(&gcm);

    int ret = mbedtls_gcm_setkey(&gcm, MBEDTLS_CIPHER_ID_AES, (const unsigned char*)key,
                                 PF_KEY_SIZE * 8);
    if (ret != 0) {
        ERROR("mbedtls_gcm_setkey failed: %d\n", ret);
        goto out;
    }

    ret = mbedtls_gcm_crypt_and_tag(&gcm, MBEDTLS_GCM_ENCRYPT, input_size, (const unsigned char*)iv,
                                    PF_IV_SIZE, aad, aad_size, input, output, PF_MAC_SIZE,
                                    (unsigned char*)mac);
    if (ret != 0) {
        ERROR("mbedtls_gcm_crypt_and_tag failed: %d\n", ret);
        goto out;
    }

    status = PF_STATUS_SUCCESS;
out:
    mbedtls_gcm_free(&gcm);
    return status;
}

pf_status_t mbedtls_aes_gcm_decrypt(const pf_key_t* key, const pf_iv_t* iv, const void* aad,
                                    size_t aad_size, const void* input, size_t input_size,
                                    void* output, const pf_mac_t* mac) {
    pf_status_t status = PF_STATUS_CALLBACK_FAILED;

    mbedtls_gcm_context gcm;
    mbedtls_gcm_init(&gcm);

    int ret = mbedtls_gcm_setkey(&gcm, MBEDTLS_CIPHER_ID_AES, (const unsigned char*)key,
                                 PF_KEY_SIZE * 8);
    if (ret != 0) {
        ERROR("mbedtls_gcm_setkey failed: %d\n", ret);
        goto out;
    }

    ret = mbedtls_gcm_auth_decrypt(&gcm, input_size, (const unsigned char*)iv, PF_IV_SIZE, aad,
                                   aad_size, (const unsigned char*)mac, PF_MAC_SIZE, input, output);
    if (ret != 0) {
        ERROR("mbedtls_gcm_auth_decrypt failed: %d\n", ret);
        goto out;
    }

    status = PF_STATUS_SUCCESS;
out:
    mbedtls_gcm_free(&gcm);
    return status;
}

static mbedtls_entropy_context g_entropy;
static mbedtls_ctr_drbg_context g_prng;

static pf_status_t mbedtls_random(uint8_t* buffer, size_t size) {
    if (mbedtls_ctr_drbg_random(&g_prng, buffer, size) != 0) {
        ERROR("Failed to get random bytes\n");
        return PF_STATUS_CALLBACK_FAILED;
    }
    return PF_STATUS_SUCCESS;
}

static int pf_set_linux_callbacks(pf_debug_f debug_f) {
    const char* prng_tag = "Gramine protected files library";

    /* Initialize mbedTLS CPRNG */
    mbedtls_entropy_init(&g_entropy);
    mbedtls_ctr_drbg_init(&g_prng);
    int ret = mbedtls_ctr_drbg_seed(&g_prng, mbedtls_entropy_func, &g_entropy,
                                    (const unsigned char*)prng_tag, strlen(prng_tag));

    if (ret != 0) {
        ERROR("Failed to initialize mbedTLS RNG: %d\n", ret);
        return -1;
    }

    pf_set_callbacks(linux_read, linux_write, linux_truncate, mbedtls_aes_cmac,
                     mbedtls_aes_gcm_encrypt, mbedtls_aes_gcm_decrypt, mbedtls_random, debug_f);
    return 0;
}

/* Debug print callback for protected files */
static void cb_debug(const char* msg) {
    DBG("%s\n", msg);
}

/* Initialize protected files for native environment */
int pf_init(void) {
    return pf_set_linux_callbacks(cb_debug);
}

/* Generate random PF key and save it to file */
int pf_generate_wrap_key(const char* wrap_key_path) {
    pf_key_t wrap_key;

    int ret = mbedtls_ctr_drbg_random(&g_prng, (unsigned char*)&wrap_key, sizeof(wrap_key));
    if (ret != 0) {
        ERROR("Failed to read random bytes: %d\n", ret);
        return ret;
    }

    if (write_file(wrap_key_path, sizeof(wrap_key), wrap_key) != 0) {
        ERROR("Failed to save wrap key\n");
        return -1;
    }

    INFO("Wrap key saved to: %s\n", wrap_key_path);
    return 0;
}

int load_wrap_key(const char* wrap_key_path, pf_key_t* wrap_key) {
    int ret = -1;
    uint64_t size = 0;
    void* buf = read_file(wrap_key_path, &size, /*buffer=*/NULL);

    if (!buf) {
        ERROR("Failed to read wrap key\n");
        goto out;
    }

    if (size != PF_KEY_SIZE) {
        ERROR("Wrap key size %zu != %zu\n", size, sizeof(*wrap_key));
        goto out;
    }

    memcpy(wrap_key, buf, sizeof(*wrap_key));
    ret = 0;

out:
    free(buf);
    return ret;
}

/* Convert a single file to the protected format */
int pf_encrypt_file(const char* input_path, const char* output_path, const pf_key_t* wrap_key) {
    int ret = -1;
    int input = -1;
    int output = -1;
    pf_context_t* pf = NULL;
    void* chunk = malloc(PF_NODE_SIZE);
    if (!chunk) {
        ERROR("Out of memory\n");
        goto out;
    }

    input = open(input_path, O_RDONLY);
    if (input < 0) {
        ERROR("Failed to open input file '%s': %s\n", input_path, strerror(errno));
        goto out;
    }

    output = open(output_path, O_RDWR | O_CREAT, PERM_rw_rw_r__);
    if (output < 0) {
        ERROR("Failed to create output file '%s': %s\n", output_path, strerror(errno));
        goto out;
    }

    INFO("Encrypting: %s -> %s\n", input_path, output_path);
    INFO("            (Gramine's sgx.protected_files must contain this exact path: \"%s\")\n",
                      output_path);

    pf_handle_t handle = (pf_handle_t)&output;
    pf_status_t pfs = pf_open(handle, output_path, /*size=*/0, PF_FILE_MODE_WRITE, /*create=*/true,
                              wrap_key, &pf);
    if (PF_FAILURE(pfs)) {
        ERROR("Failed to open output PF: %s\n", pf_strerror(pfs));
        goto out;
    }

    /* Process file contents */
    uint64_t input_size = get_file_size(input);
    if (input_size == (uint64_t)-1) {
        ERROR("Failed to get size of input file '%s': %s\n", input_path, strerror(errno));
        goto out;
    }

    uint64_t input_offset = 0;

    while (true) {
        ssize_t chunk_size = read(input, chunk, PF_NODE_SIZE);
        if (chunk_size == 0) // EOF
            break;

        if (chunk_size < 0) {
            if (errno == -EINTR)
                continue;

            ERROR("Failed to read file '%s': %s\n", input_path, strerror(errno));
            goto out;
        }

        pfs = pf_write(pf, input_offset, chunk_size, chunk);
        if (PF_FAILURE(pfs)) {
            ERROR("Failed to write to output PF: %s\n", pf_strerror(pfs));
            goto out;
        }

        input_offset += chunk_size;
    }

    ret = 0;

out:
    if (pf) {
        if (PF_FAILURE(pf_close(pf))) {
            ERROR("failed to close PF\n");
            ret = -1;
        }
    }

    free(chunk);
    if (input >= 0)
        close(input);
    if (output >= 0)
        close(output);
    return ret;
}

/* Convert a single file from the protected format */
int pf_decrypt_file(const char* input_path, const char* output_path, bool verify_path,
                    const pf_key_t* wrap_key) {
    int ret = -1;
    int input = -1;
    int output = -1;
    pf_context_t* pf = NULL;
    void* chunk = malloc(PF_NODE_SIZE);
    if (!chunk) {
        ERROR("Out of memory\n");
        goto out;
    }

    input = open(input_path, O_RDONLY);
    if (input < 0) {
        ERROR("Failed to open input file '%s': %s\n", input_path, strerror(errno));
        goto out;
    }

    output = open(output_path, O_RDWR | O_CREAT, PERM_rw_rw_r__);
    if (output < 0) {
        ERROR("Failed to create output file '%s': %s\n", output_path, strerror(errno));
        goto out;
    }

    INFO("Decrypting: %s -> %s\n", input_path, output_path);

    /* Get underlying file size */
    uint64_t input_size = get_file_size(input);
    if (input_size == (uint64_t)-1) {
        ERROR("Failed to get size of input file '%s': %s\n", input_path, strerror(errno));
        goto out;
    }

    const char* path = verify_path ? input_path : NULL;
    pf_status_t pfs = pf_open((pf_handle_t)&input, path, input_size, PF_FILE_MODE_READ,
                              /*create=*/false, wrap_key, &pf);
    if (PF_FAILURE(pfs)) {
        ERROR("Opening protected input file failed: %s\n", pf_strerror(pfs));
        goto out;
    }

    /* Process file contents */
    uint64_t data_size;
    pfs = pf_get_size(pf, &data_size);
    if (PF_FAILURE(pfs)) {
        ERROR("pf_get_size failed: %s\n", pf_strerror(pfs));
        goto out;
    }

    if (ftruncate64(output, data_size) < 0) {
        ERROR("ftruncate64 output file '%s' failed: %s\n", output_path, strerror(errno));
        goto out;
    }

    uint64_t input_offset = 0;

    while (true) {
        assert(input_offset <= data_size);
        uint64_t chunk_size = MIN(data_size - input_offset, PF_NODE_SIZE);
        if (chunk_size == 0)
            break;

        size_t bytes_read = 0;
        pfs = pf_read(pf, input_offset, chunk_size, chunk, &bytes_read);
        if (bytes_read != chunk_size) {
            pfs = PF_STATUS_CORRUPTED;
        }
        if (PF_FAILURE(pfs)) {
            ERROR("Read from protected file failed (offset %" PRIu64 ", size %" PRIu64 "): %s\n",
                  input_offset, chunk_size, pf_strerror(pfs));
            goto out;
        }

        ssize_t written = write(output, chunk, chunk_size);

        if (written < 0) {
            if (errno == -EINTR)
                continue;

            ERROR("Failed to write file '%s': %s\n", output_path, strerror(errno));
            goto out;
        }

        input_offset += written;
    }

    ret = 0;

out:
    free(chunk);
    if (pf)
        pf_close(pf);
    if (input >= 0)
        close(input);
    if (output >= 0)
        close(output);
    return ret;
}

enum processing_mode_t {
    MODE_ENCRYPT = 1,
    MODE_DECRYPT = 2,
};

static int process_files(const char* input_dir, const char* output_dir, const char* wrap_key_path,
                         enum processing_mode_t mode, bool verify_path) {
    int ret = -1;
    pf_key_t wrap_key;
    struct stat st;
    char* input_path  = NULL;
    char* output_path = NULL;

    if (mode != MODE_ENCRYPT && mode != MODE_DECRYPT) {
        ERROR("Invalid mode: %d\n", mode);
        goto out;
    }

    if (mode == MODE_ENCRYPT && verify_path) {
        ERROR("Path verification can't be on in MODE_ENCRYPT\n");
        goto out;
    }

    ret = load_wrap_key(wrap_key_path, &wrap_key);
    if (ret != 0)
        goto out;

    if (stat(input_dir, &st) != 0) {
        ERROR("Failed to stat input path %s: %s\n", input_dir, strerror(errno));
        goto out;
    }

    /* single file? */
    if (S_ISREG(st.st_mode)) {
        if (mode == MODE_ENCRYPT)
            return pf_encrypt_file(input_dir, output_dir, &wrap_key);
        else
            return pf_decrypt_file(input_dir, output_dir, verify_path, &wrap_key);
    }

    ret = mkdir(output_dir, PERM_rwxrwxr_x);
    if (ret != 0 && errno != EEXIST) {
        ERROR("Failed to create directory %s: %s\n", output_dir, strerror(errno));
        goto out;
    }

    /* Process input directory */
    struct dirent* dir;
    DIR* dfd = opendir(input_dir);
    if (!dfd) {
        ERROR("Failed to open input directory: %s\n", strerror(errno));
        goto out;
    }

    size_t input_path_size, output_path_size;
    while ((dir = readdir(dfd)) != NULL) {
        if (!strcmp(dir->d_name, "."))
            continue;
        if (!strcmp(dir->d_name, ".."))
            continue;

        input_path_size = strlen(input_dir) + 1 + strlen(dir->d_name) + 1;
        output_path_size = strlen(output_dir) + 1 + strlen(dir->d_name) + 1;

        input_path = malloc(input_path_size);
        if (!input_path) {
            ERROR("No memory\n");
            goto out;
        }

        output_path = malloc(output_path_size);
        if (!output_path) {
            ERROR("No memory\n");
            goto out;
        }

        snprintf(input_path, input_path_size, "%s/%s", input_dir, dir->d_name);
        snprintf(output_path, output_path_size, "%s/%s", output_dir, dir->d_name);

        if (stat(input_path, &st) != 0) {
            ERROR("Failed to stat input file %s: %s\n", input_path, strerror(errno));
            goto out;
        }

        if (S_ISREG(st.st_mode)) {
            if (mode == MODE_ENCRYPT)
                ret = pf_encrypt_file(input_path, output_path, &wrap_key);
            else
                ret = pf_decrypt_file(input_path, output_path, verify_path, &wrap_key);

            if (ret != 0)
                goto out;
        } else if (S_ISDIR(st.st_mode)) {
            /* process directory recursively */
            ret = process_files(input_path, output_path, wrap_key_path, mode, verify_path);
            if (ret != 0)
                goto out;
        } else {
            INFO("Skipping non-regular file %s\n", input_path);
        }

        free(input_path);
        input_path = NULL;
        free(output_path);
        output_path = NULL;
    }
    ret = 0;

out:
    free(input_path);
    free(output_path);
    return ret;
}

/* Convert a file or directory (recursively) to the protected format */
int pf_encrypt_files(const char* input_dir, const char* output_dir, const char* wrap_key_path) {
    return process_files(input_dir, output_dir, wrap_key_path, MODE_ENCRYPT, false);
}

/* Convert a file or directory (recursively) from the protected format */
int pf_decrypt_files(const char* input_dir, const char* output_dir, bool verify_path,
                     const char* wrap_key_path) {
    return process_files(input_dir, output_dir, wrap_key_path, MODE_DECRYPT, verify_path);
}
