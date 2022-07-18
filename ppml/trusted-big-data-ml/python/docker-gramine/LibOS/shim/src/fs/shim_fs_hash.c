/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains functions to generate hash values for FS paths.
 */

#include "shim_fs.h"

HASHTYPE hash_str(const char* p) {
    HASHTYPE hash = 0;
    HASHTYPE tmp;

    size_t len = strlen(p);

    for (; len >= sizeof(hash); p += sizeof(hash), len -= sizeof(hash)) {
        memcpy(&tmp, p, sizeof(tmp)); /* avoid pointer alignment issues */
        hash += tmp;
        hash *= 9;
    }

    if (len) {
        HASHTYPE rest = 0;
        for (; len > 0; p++, len--) {
            rest <<= 8;
            rest += (HASHTYPE)*p;
        }
        hash += rest;
        hash *= 9;
    }

    return hash;
}

HASHTYPE hash_name(HASHTYPE parent_hbuf, const char* name) {
    return (parent_hbuf + hash_str(name)) * 9;
}

HASHTYPE hash_abs_path(struct shim_dentry* dent) {
    HASHTYPE digest = 0;

    while (true) {
        struct shim_dentry* up = dentry_up(dent);
        if (!up)
            break;

        digest += hash_str(dent->name);
        digest *= 9;
        dent = up;
    }
    return digest;
}
