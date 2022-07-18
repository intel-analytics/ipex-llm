/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Dmitrii Kuvaiskii <dmitrii.kuvaiskii@intel.com>
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 */

#include <stdint.h>

#include "api.h"
#include "assert.h"
#include "toml.h"
#include "toml_utils.h"

/* returns a pointer to next occurrence of `ch` in `s`, or null byte ending the string if it wasn't
 * found */
static char* find_next_char(char* s, char ch) {
    while (*s && *s != ch)
        s++;
    return s;
}

/* Searches for a dotted-key (e.g. "fs.root.uri") from `root`; returns NULL if value for such key is
 * not found. Double quotes are respected, same as in TOML. */
static toml_raw_t toml_raw_in_dottedkey(const toml_table_t* root, const char* _key) {
    char* key = strdup(_key);
    if (!key)
        return NULL;

    toml_raw_t res = NULL;

    assert(root);
    const toml_table_t* cur_table = root;

    char* subkey = key;
    while (*subkey) {
        char* subkey_end;
        if (*subkey == '"') {
            // quoted subkey
            subkey++;
            subkey_end = find_next_char(subkey, '"');
            if (subkey_end[0] != '"'  || (subkey_end[1] != '.' && subkey_end[1] != '\0'))
                goto out; // incorrectly terminated '"'
            *subkey_end = '\0';
            subkey_end++; // points to '.' or '\0' now
        } else {
            // unquoted subkey
            subkey_end = find_next_char(subkey, '.');
        }
        if (!*subkey_end) {
            // this is the last subkey, jump out and parse it using `toml_raw_in`
            break;
        }
        // there will be more subkeys afterwards
        *subkey_end = '\0';
        cur_table = toml_table_in(cur_table, subkey);
        if (!cur_table)
            goto out;
        subkey = subkey_end + 1;
    }

    res = toml_raw_in(cur_table, subkey);
out:
    free(key);
    return res;
}

bool toml_key_exists(const toml_table_t* root, const char* key) {
    toml_raw_t raw = toml_raw_in_dottedkey(root, key);
    return !!raw;
}

int toml_bool_in(const toml_table_t* root, const char* key, bool defaultval, bool* retval) {
    toml_raw_t raw = toml_raw_in_dottedkey(root, key);
    if (!raw) {
        *retval = defaultval;
        return 0;
    }

    int intval;
    int ret = toml_rtob(raw, &intval);
    if (ret != 0)
        return -1;

    *retval = (bool)intval;
    return 0;
}

int toml_int_in(const toml_table_t* root, const char* key, int64_t defaultval, int64_t* retval) {
    toml_raw_t raw = toml_raw_in_dottedkey(root, key);
    if (!raw) {
        *retval = defaultval;
        return 0;
    }
    return toml_rtoi(raw, retval);
}

int toml_string_in(const toml_table_t* root, const char* key, char** retval) {
    toml_raw_t raw = toml_raw_in_dottedkey(root, key);
    if (!raw) {
        *retval = NULL;
        return 0;
    }
    return toml_rtos(raw, retval);
}

int toml_sizestring_in(const toml_table_t* root, const char* key, uint64_t defaultval,
                       uint64_t* retval) {
    toml_raw_t raw = toml_raw_in_dottedkey(root, key);
    if (!raw) {
        *retval = defaultval;
        return 0;
    }

    char* str = NULL;
    if (toml_rtos(raw, &str) < 0) {
        return -1;
    }
    assert(str);

    uint64_t size;
    int ret = parse_size_str(str, &size);
    free(str);

    if (ret < 0)
        return -1;

    *retval = size;
    return 0;
}
