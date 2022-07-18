/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 * Copyright (C) 2019 Intel Corporation
 */

/* See README.rst for protected files overview */

#ifndef PROTECTED_FILES_H_
#define PROTECTED_FILES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*! Size of the AES-GCM encryption key */
#define PF_KEY_SIZE 16

/*! Size of IV for AES-GCM */
#define PF_IV_SIZE 12

/*! Size of MAC fields */
#define PF_MAC_SIZE 16

typedef uint8_t pf_iv_t[PF_IV_SIZE];
typedef uint8_t pf_mac_t[PF_MAC_SIZE];
typedef uint8_t pf_key_t[PF_KEY_SIZE];
typedef uint8_t pf_keyid_t[32]; /* key derivation material */

extern pf_key_t g_pf_mrenclave_key;
extern pf_key_t g_pf_mrsigner_key;
extern pf_key_t g_pf_wrap_key;
extern bool g_pf_wrap_key_set;

typedef enum _pf_status_t {
    PF_STATUS_SUCCESS              = 0,
    PF_STATUS_UNKNOWN_ERROR        = -1,
    PF_STATUS_UNINITIALIZED        = -2,
    PF_STATUS_INVALID_PARAMETER    = -3,
    PF_STATUS_INVALID_MODE         = -4,
    PF_STATUS_NO_MEMORY            = -5,
    PF_STATUS_INVALID_VERSION      = -6,
    PF_STATUS_INVALID_HEADER       = -7,
    PF_STATUS_INVALID_PATH         = -8,
    PF_STATUS_MAC_MISMATCH         = -9,
    PF_STATUS_NOT_IMPLEMENTED      = -10,
    PF_STATUS_CALLBACK_FAILED      = -11,
    PF_STATUS_PATH_TOO_LONG        = -12,
    PF_STATUS_RECOVERY_NEEDED      = -13,
    PF_STATUS_FLUSH_ERROR          = -14,
    PF_STATUS_CRYPTO_ERROR         = -15,
    PF_STATUS_CORRUPTED            = -16,
    PF_STATUS_WRITE_TO_DISK_FAILED = -17,
} pf_status_t;

#define PF_SUCCESS(status) ((status) == PF_STATUS_SUCCESS)
#define PF_FAILURE(status) ((status) != PF_STATUS_SUCCESS)

#define PF_NODE_SIZE 4096U

/*! PF open modes */
typedef enum _pf_file_mode_t {
    PF_FILE_MODE_READ  = 1,
    PF_FILE_MODE_WRITE = 2,
} pf_file_mode_t;

/*! Opaque file handle type, interpreted by callbacks as necessary */
typedef void* pf_handle_t;

/*!
 * \brief File read callback
 *
 * \param [in] handle File handle
 * \param [out] buffer Buffer to read to
 * \param [in] offset Offset to read from
 * \param [in] size Number of bytes to read
 * \return PF status
 */
typedef pf_status_t (*pf_read_f)(pf_handle_t handle, void* buffer, uint64_t offset, size_t size);

/*!
 * \brief File write callback
 *
 * \param [in] handle File handle
 * \param [in] buffer Buffer to write from
 * \param [in] offset Offset to write to
 * \param [in] size Number of bytes to write
 * \return PF status
 */
typedef pf_status_t (*pf_write_f)(pf_handle_t handle, const void* buffer, uint64_t offset,
                                  size_t size);

/*!
 * \brief File truncate callback
 *
 * \param [in] handle File handle
 * \param [in] size Target file size
 * \return PF status
 */
typedef pf_status_t (*pf_truncate_f)(pf_handle_t handle, uint64_t size);

/*!
 * \brief Debug print callback
 *
 * \param [in] msg Message to print
 */
typedef void (*pf_debug_f)(const char* msg);

/*!
 * \brief AES-CMAC callback used for key derivation
 *
 * \param [in] key AES-GCM key
 * \param [in] input Plaintext data
 * \param [in] input_size Size of \a input in bytes
 * \param [out] mac MAC computed for \a input
 * \return PF status
 */
typedef pf_status_t (*pf_aes_cmac_f)(const pf_key_t* key, const void* input, size_t input_size,
                                     pf_mac_t* mac);

/*!
 * \brief AES-GCM encrypt callback
 *
 * \param [in] key AES-GCM key
 * \param [in] iv Initialization vector
 * \param [in] aad (optional) Additional authenticated data
 * \param [in] aad_size Size of \a aad in bytes
 * \param [in] input Plaintext data
 * \param [in] input_size Size of \a input in bytes
 * \param [out] output Buffer for encrypted data (size: \a input_size)
 * \param [out] mac MAC computed for \a input and \a aad
 * \return PF status
 */
typedef pf_status_t (*pf_aes_gcm_encrypt_f)(const pf_key_t* key, const pf_iv_t* iv, const void* aad,
                                            size_t aad_size, const void* input, size_t input_size,
                                            void* output, pf_mac_t* mac);

/*!
 * \brief AES-GCM decrypt callback
 *
 * \param [in] key AES-GCM key
 * \param [in] iv Initialization vector
 * \param [in] aad (optional) Additional authenticated data
 * \param [in] aad_size Size of \a aad in bytes
 * \param [in] input Encrypted data
 * \param [in] input_size Size of \a input in bytes
 * \param [out] output Buffer for decrypted data (size: \a input_size)
 * \param [in] mac Expected MAC
 * \return PF status
 */
typedef pf_status_t (*pf_aes_gcm_decrypt_f)(const pf_key_t* key, const pf_iv_t* iv, const void* aad,
                                            size_t aad_size, const void* input, size_t input_size,
                                            void* output, const pf_mac_t* mac);

/*!
 * \brief Cryptographic random number generator callback
 *
 * \param [out] buffer Buffer to fill with random bytes
 * \param [in] size Size of \a buffer in bytes
 * \return PF status
 */
typedef pf_status_t (*pf_random_f)(uint8_t* buffer, size_t size);

/*!
 * \brief Initialize I/O callbacks
 *
 * \param [in] read_f File read callback
 * \param [in] write_f File write callback
 * \param [in] truncate_f File truncate callback
 * \param [in] aes_cmac_f AES-CMAC callback
 * \param [in] aes_gcm_encrypt_f AES-GCM encrypt callback
 * \param [in] aes_gcm_decrypt_f AES-GCM decrypt callback
 * \param [in] random_f Cryptographic random number generator callback
 * \param [in] debug_f (optional) Debug print callback
 *
 * \details Must be called before any actual APIs
 */
void pf_set_callbacks(pf_read_f read_f, pf_write_f write_f, pf_truncate_f truncate_f,
                      pf_aes_cmac_f aes_cmac_f, pf_aes_gcm_encrypt_f aes_gcm_encrypt_f,
                      pf_aes_gcm_decrypt_f aes_gcm_decrypt_f, pf_random_f random_f,
                      pf_debug_f debug_f);

/*! Context representing an open protected file */
typedef struct pf_context pf_context_t;

/* Public API */

/*!
 * \brief Convert error code to error message
 *
 * \param [in] err Error code
 * \return Error message
 */
const char* pf_strerror(int err);

/*!
 * \brief Open a protected file
 *
 * \param [in] handle Open underlying file handle
 * \param [in] path Path to the file. If NULL and \a create is false, don't check path for validity.
 * \param [in] underlying_size Underlying file size
 * \param [in] mode Access mode
 * \param [in] create Overwrite file contents if true
 * \param [in] key Wrap key
 * \param [out] context PF context for later calls
 * \return PF status
 */
pf_status_t pf_open(pf_handle_t handle, const char* path, uint64_t underlying_size,
                    pf_file_mode_t mode, bool create, const pf_key_t* key, pf_context_t** context);

/*!
 * \brief Close a protected file and commit all changes to disk
 *
 * \param [in] pf PF context
 * \return PF status
 */
pf_status_t pf_close(pf_context_t* pf);

/*!
 * \brief Read from a protected file
 *
 * \param [in] pf PF context
 * \param [in] offset Data offset to read from
 * \param [in] size Number of bytes to read
 * \param [out] output Destination buffer
 * \param [out] bytes_read Number of bytes actually read
 * \return PF status
 */
pf_status_t pf_read(pf_context_t* pf, uint64_t offset, size_t size, void* output,
                    size_t* bytes_read);

/*!
 * \brief Write to a protected file
 *
 * \param [in] pf PF context
 * \param [in] offset Data offset to write to
 * \param [in] size Number of bytes to write
 * \param [in] input Source buffer
 * \return PF status
 */
pf_status_t pf_write(pf_context_t* pf, uint64_t offset, size_t size, const void* input);

/*!
 * \brief Get data size of a PF
 *
 * \param [in] pf PF context
 * \param [out] size Data size of \a pf
 * \return PF status
 */
pf_status_t pf_get_size(pf_context_t* pf, uint64_t* size);

/*!
 * \brief Set data size of a PF
 *
 * \param [in] pf PF context
 * \param [in] size Data size to set
 * \return PF status
 * \details If the file is extended, added bytes are zero.
 *          Truncation is not implemented yet (TODO).
 */
pf_status_t pf_set_size(pf_context_t* pf, uint64_t size);

/*!
 * \brief Rename a PF.
 *
 * \param pf        PF context.
 * \param new_path  New file path.
 *
 * Updates the path inside protected file header, and flushes all changes. The caller is responsible
 * for renaming the underlying file.
 */
pf_status_t pf_rename(pf_context_t* pf, const char* new_path);

/*!
 * \brief Get underlying handle of a PF
 *
 * \param [in] pf PF context
 * \param [out] handle Handle to the backing file
 * \return PF status
 */
pf_status_t pf_get_handle(pf_context_t* pf, pf_handle_t* handle);

/*!
 * \brief Flush any pending data of a protected file to disk
 *
 * \param [in] pf PF context
 * \return PF status
 */
pf_status_t pf_flush(pf_context_t* pf);

#endif /* PROTECTED_FILES_H_ */
