/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * This module implements encrypted files. It is a wrapper around the platform-independent
 * `protected_files` module, and PAL handles.
 *
 * NOTE: There is currently no notion of file permissions, all files are open in read-write mode.
 */

#ifndef SHIM_FS_ENCRYPTED_
#define SHIM_FS_ENCRYPTED_

#include <stddef.h>

#include "list.h"
#include "pal.h"
#include "protected_files.h"
#include "shim_types.h"

/*
 * Represents a named key for opening files. The key might not be set yet: value of a key can be
 * specified in the manifest, or set using `update_encrypted_files_key`. Before the key is set,
 * operations that use it will fail.
 */
DEFINE_LIST(shim_encrypted_files_key);
DEFINE_LISTP(shim_encrypted_files_key);
struct shim_encrypted_files_key {
    char* name;
    bool is_set;
    pf_key_t pf_key;

    LIST_TYPE(shim_encrypted_files_key) list;
};

/*
 * Represents a specific encrypted file. The file is open as long as `use_count` is greater than 0.
 * Note that the file can be open and closed multiple times before it's destroyed.
 *
 * Operations on a single `shim_encrypted_file` are NOT thread-safe, it is intended to be protected
 * by a lock.
 */
struct shim_encrypted_file {
    size_t use_count;
    char* uri;
    struct shim_encrypted_files_key* key;

    /* `pf` and `pal_handle` are non-null as long as `use_count` is greater than 0 */
    pf_context_t* pf;
    PAL_HANDLE pal_handle;
};

/*
 * \brief Initialize the encrypted files module.
 *
 * Performs necessary setup, including loading keys specified in manifest.
 */
int init_encrypted_files(void);

/*
 * \brief Retrieve a key.
 *
 * Returns a key with a given name, or NULL if it has not been created yet. Note that even if the
 * key exists, it might not be set yet (see `struct shim_encrypted_files_key`).
 *
 * This does not pass ownership of the key: the key objects are still managed by this module.
 */
struct shim_encrypted_files_key* get_encrypted_files_key(const char* name);

/*
 * \brief List existing keys.
 *
 * Calls `callback` on each currently existing key.
 */
int list_encrypted_files_keys(int (*callback)(struct shim_encrypted_files_key* key, void* arg),
                              void* arg);

/*
 * \brief Retrieve or create a key.
 *
 * Sets `*out_key` to a key with given name. If the key has not been created yet, creates a new one.
 *
 * Similar to `get_encrypted_files_key`, this does not pass ownership of `*out_key`.
 */
int get_or_create_encrypted_files_key(const char* name, struct shim_encrypted_files_key** out_key);

/*
 * \brief Read value of given key.
 *
 * \param      key     The key to read.
 * \param[out] pf_key  On success, will be set to the current value.
 *
 * \returns `true` if the key has a value, `false` otherwise
 *
 * If the key has already been set, writes its value to `*pf_key` and returns `true`. Otherwise,
 * returns `false`.
 */
bool read_encrypted_files_key(struct shim_encrypted_files_key* key, pf_key_t* pf_key);

/*
 * \brief Update value of given key.
 *
 * \param key     The key to update.
 * \param pf_key  New value for the key.
 */
void update_encrypted_files_key(struct shim_encrypted_files_key* key, const pf_key_t* pf_key);

/*
 * \brief Open an existing encrypted file.
 *
 * \param      uri      PAL URI to open, has to begin with "file:".
 * \param      key      Key, has to be already set.
 * \param[out] out_enc  On success, set to a newly created `shim_encrypted_file` object.
 *
 * `uri` has to correspond to an existing file that can be decrypted with `key`.
 *
 * The newly created `shim_encrypted_file` object will have `use_count` set to 1.
 */
int encrypted_file_open(const char* uri, struct shim_encrypted_files_key* key,
                        struct shim_encrypted_file** out_enc);

/*
 * \brief Create a new encrypted file.
 *
 * \param      uri      PAL URI to open, has to begin with "file:".
 * \param      perm     Permissions for the new file.
 * \param      key      Key, has to be already set.
 * \param[out] out_enc  On success, set to a newly created `shim_encrypted_file` object.
 *
 * `uri` must not correspond to an existing file.
 *
 * The newly created `shim_encrypted_file` object will have `use_count` set to 1.
 */
int encrypted_file_create(const char* uri, mode_t perm, struct shim_encrypted_files_key* key,
                          struct shim_encrypted_file** out_enc);

/*
 * \brief Deallocate an encrypted file.
 *
 * `enc` needs to have `use_count` set to 0.
 */
void encrypted_file_destroy(struct shim_encrypted_file* enc);

/*
 * \brief Increase the use count of an encrypted file.
 *
 * This increases `use_count`, and opens the file if `use_count` was 0.
 */
int encrypted_file_get(struct shim_encrypted_file* enc);

/*
 * \brief Decrease the use count of an encrypted file.
 *
 * This decreases `use_count`, and closes the file if it reaches 0.
 */
void encrypted_file_put(struct shim_encrypted_file* enc);

/*
 * \brief Flush pending writes to an encrypted file.
 */
int encrypted_file_flush(struct shim_encrypted_file* enc);

int encrypted_file_read(struct shim_encrypted_file* enc, void* buf, size_t buf_size,
                        file_off_t offset, size_t* out_count);
int encrypted_file_write(struct shim_encrypted_file* enc, const void* buf, size_t buf_size,
                         file_off_t offset, size_t* out_count);
int encrypted_file_rename(struct shim_encrypted_file* enc, const char* new_uri);

int encrypted_file_get_size(struct shim_encrypted_file* enc, file_off_t* out_size);
int encrypted_file_set_size(struct shim_encrypted_file* enc, file_off_t size);

int parse_pf_key(const char* key_str, pf_key_t* pf_key);

/* TODO: This function is used only by a feature deprecated in v1.2, remove two versions later. */
int dump_pf_key(const pf_key_t* pf_key, char* buf, size_t buf_size);

#endif /* SHIM_FS_ENCRYPTED_ */
