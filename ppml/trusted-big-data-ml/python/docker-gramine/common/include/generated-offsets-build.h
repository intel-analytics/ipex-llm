/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Support for generating constants (structure offsets etc.) from C code, for use in assembly and
 * Python.
 *
 * To use, create a C file with `generated_offsets` array (ending with `OFFSET_END`) and
 * `generated_offsets_name` string. Then, build an executable from `generated-offsets-print.c` and
 * your C file.
 *
 * TODO: The name "offsets" is not accurate anymore, we also use this setup for other values (such
 * as sizes or masks). Consider changing it.
 */

#ifndef GENERATED_OFFSETS_BUILD_H
#define GENERATED_OFFSETS_BUILD_H

#include <stdint.h>

struct generated_offset {
    const char* name;
    uint64_t value;
};

extern const struct generated_offset generated_offsets[];
extern const char* generated_offsets_name;

#define DEFINE(name, value) { #name, value }

#define OFFSET(name, str, member)     DEFINE(name, offsetof(struct str, member))
#define OFFSET_T(name, str_t, member) DEFINE(name, offsetof(str_t, member))

#define OFFSET_END { NULL, 0 }

#endif /* GENERATED_OFFSETS_BUILD_H */
