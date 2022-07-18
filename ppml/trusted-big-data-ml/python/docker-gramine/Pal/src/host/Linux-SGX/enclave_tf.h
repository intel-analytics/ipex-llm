/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation */

/* Trusted files (TF) are integrity protected and transparently verified when accessed by Gramine
 * or by app running inside Gramine. For each file that requires authentication (specified in the
 * manifest as "sgx.trusted_files"), a SHA256 hash is generated and stored in the manifest, signed
 * and verified as part of the enclave's crypto measurement. When user opens such a file, Gramine
 * loads the whole file, calculates its SHA256 hash, and checks against the corresponding hash in
 * the manifest. If the hashes do not match, the file access will be rejected.
 *
 * During the generation of the SHA256 hash, a 128-bit hash (truncated SHA256) is also generated for
 * each chunk (of size TRUSTED_CHUNK_SIZE) in the file. The per-chunk hashes are used for partial
 * verification in future reads, to avoid re-verifying the whole file again or the need of caching
 * file contents.
 */

/* TODO: Move trusted/allowed files implementation into a separate file (`enclave_tf.c`?) */

#ifndef ENCLAVE_TF_H_
#define ENCLAVE_TF_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "api.h"
#include "enclave_tf_structs.h"
#include "pal.h"

int init_seal_key_material(void);

int init_file_check_policy(void);
int get_file_check_policy(void);

/*!
 * \brief Get trusted/allowed file struct, if corresponding path entry exists in the manifest
 *
 * \param path  Normalized path to search for trusted/allowed files
 *
 * \return trusted/allowed file struct if found, NULL otherwise
 */
struct trusted_file* get_trusted_or_allowed_file(const char* path);

/*!
 * \brief Open the file as trusted or allowed, according to the manifest
 *
 * \param tf                trusted file struct corresponding to this file
 * \param file              file handle to be opened
 * \param create            whether this file is newly created
 * \param out_chunk_hashes  array of hashes over file chunks
 * \param out_size          returns size of opened file
 * \param out_umem          untrusted memory address at which the file was loaded
 *
 * \return 0 on success, negative error code on failure
 */
int load_trusted_or_allowed_file(struct trusted_file* tf, PAL_HANDLE file, bool create,
                                 sgx_chunk_hash_t** out_chunk_hashes, uint64_t* out_size,
                                 void** out_umem);

/*!
 * \brief Copy and check file contents from untrusted outside buffer to in-enclave buffer
 *
 * \param path            file path (currently only for a log message)
 * \param buf             in-enclave buffer where contents of the file are copied
 * \param umem            start of untrusted file memory mapped outside the enclave
 * \param aligned_offset  offset into file contents to copy, aligned to TRUSTED_CHUNK_SIZE
 * \param aligned_end     end of file contents to copy, aligned to TRUSTED_CHUNK_SIZE
 * \param offset          unaligned offset into file contents to copy
 * \param end             unaligned end of file contents to copy
 * \param chunk_hashes    array of hashes of all file chunks
 * \param file_size       total size of the file
 *
 * \return 0 on success, negative error code on failure
 */
int copy_and_verify_trusted_file(const char* path, uint8_t* buf, const void* umem,
                                 off_t aligned_offset, off_t aligned_end, off_t offset, off_t end,
                                 sgx_chunk_hash_t* chunk_hashes, size_t file_size);

int init_trusted_files(void);
int init_allowed_files(void);

#endif /* ENCLAVE_TF_H_ */
