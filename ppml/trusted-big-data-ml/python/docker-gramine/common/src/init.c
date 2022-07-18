/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#include "init.h"

/*
 * Helper symbols for accessing the `.init_array` section at run time. These are part of default
 * linker script (you can see it by running `ld --verbose`) as well as our custom linker scripts.
 *
 * NOTE: We rely on the fact that each ELF object (PAL, LibOS) contains its own copy of this module,
 * referring to `__init_array_start` and `__init_array_end` in that object.
 */
extern void (*__init_array_start)(void);
extern void (*__init_array_end)(void);

void call_init_array(void) {
    void (**func)(void);
    for (func = &__init_array_start; func < &__init_array_end; func++) {
        (*func)();
    }
}
