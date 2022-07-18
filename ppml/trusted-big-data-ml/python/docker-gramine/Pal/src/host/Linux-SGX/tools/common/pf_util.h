/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 */

#ifndef PF_UTIL_H
#define PF_UTIL_H

#include "protected_files.h"

/* High-level protected files helper functions */

/*! Initialize protected files for native environment */
int pf_init(void);

/*! Generate random PF key and save it to file */
int pf_generate_wrap_key(const char* wrap_key_path);

/*! Convert a single file to the protected format */
int pf_encrypt_file(const char* input_path, const char* output_path, const pf_key_t* wrap_key);

/*! Convert a single file from the protected format */
int pf_decrypt_file(const char* input_path, const char* output_path, bool verify_path,
                    const pf_key_t* wrap_key);

/*! Convert a file or directory (recursively) to the protected format */
int pf_encrypt_files(const char* input_dir, const char* output_dir, const char* wrap_key_path);

/*! Convert a file or directory (recursively) from the protected format */
int pf_decrypt_files(const char* input_dir, const char* output_dir, bool verify_path,
                     const char* wrap_key_path);

/*! AES-CMAC */
pf_status_t mbedtls_aes_cmac(const pf_key_t* key, const void* input, size_t input_size,
                             pf_mac_t* mac);

/*! AES-GCM encrypt */
pf_status_t mbedtls_aes_gcm_encrypt(const pf_key_t* key, const pf_iv_t* iv, const void* aad,
                                    size_t aad_size, const void* input, size_t input_size,
                                    void* output, pf_mac_t* mac);

/*! AES-GCM decrypt */
pf_status_t mbedtls_aes_gcm_decrypt(const pf_key_t* key, const pf_iv_t* iv, const void* aad,
                                    size_t aad_size, const void* input, size_t input_size,
                                    void* output, const pf_mac_t* mac);

/*! Load PF wrap key from file */
int load_wrap_key(const char* wrap_key_path, pf_key_t* wrap_key);

#endif
