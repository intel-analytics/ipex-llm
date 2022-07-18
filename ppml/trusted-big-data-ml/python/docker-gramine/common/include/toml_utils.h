/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#ifndef TOML_UTILS_H_
#define TOML_UTILS_H_

#include "toml.h"

/*!
 * \brief Check if a key was specified in TOML manifest.
 *
 * \param root  Root table of the TOML manifest.
 * \param key   Dotted key (e.g. "loader.insecure__use_cmdline_argv").
 */
bool toml_key_exists(const toml_table_t* root, const char* key);

/*!
 * \brief Find a bool key-value in TOML manifest.
 *
 * \param root        Root table of the TOML manifest.
 * \param key         Dotted key (e.g. "loader.insecure__use_cmdline_argv").
 * \param defaultval  `retval` is set to this value if not found in the manifest.
 * \param retval      Pointer to output bool.
 *
 * \returns 0 if there were no errors (but value may have not been found in manifest and was set to
 *          default one) or negative if there were errors during conversion to bool.
 */
int toml_bool_in(const toml_table_t* root, const char* key, bool defaultval, bool* retval);

/*!
 * \brief Find an integer key-value in TOML manifest.
 *
 * \param root        Root table of the TOML manifest.
 * \param key         Dotted key (e.g. "sgx.thread_num").
 * \param defaultval  `retval` is set to this value if not found in the manifest.
 * \param retval      Pointer to output integer.
 *
 * \returns 0 if there were no errors (but value may have not been found in manifest and was set to
 *          default one) or negative if there were errors during conversion to int.
 */
int toml_int_in(const toml_table_t* root, const char* key, int64_t defaultval, int64_t* retval);

/*!
 * \brief Find a string key-value in TOML manifest.
 *
 * \param root    Root table of the TOML manifest.
 * \param key     Dotted key (e.g. "fs.root.uri").
 * \param retval  Pointer to output string.
 *
 * \returns 0 if there were no errors (but value may have not been found in manifest and was set to
 *          NULL) or negative if there were errors during conversion to string.
 */
int toml_string_in(const toml_table_t* root, const char* key, char** retval);

/*!
 * \brief Find a "size" string key-value in TOML manifest (parsed via `parse_size_str()`).
 *
 * \param root        Root table of the TOML manifest.
 * \param key         Dotted key (e.g. "sys.stack.size").
 * \param defaultval  `retval` is set to this value if not found in the manifest.
 * \param retval      Pointer to output integer.
 *
 * \returns 0 if there were no errors (but value may have not been found in manifest and was set to
 *          default one) or negative if there were errors during conversion to "size" string.
 */
int toml_sizestring_in(const toml_table_t* root, const char* key, uint64_t defaultval,
                       uint64_t* retval);

#endif /* TOML_UTILS_H_ */
