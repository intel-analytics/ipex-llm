/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#include "api.h"

char* strchr(const char* s, int c) {
    while (true) {
        if (*s == c)
            return (char*)s;
        if (*s == '\0')
            return NULL;
        s++;
    }
}
