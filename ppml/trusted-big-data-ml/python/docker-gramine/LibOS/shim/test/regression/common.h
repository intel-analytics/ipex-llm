/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#ifndef COMMON_H_
#define COMMON_H_

#define OVERFLOWS(type, val)                        \
    ({                                              \
        type __dummy;                               \
        __builtin_add_overflow((val), 0, &__dummy); \
    })

#endif /* COMMON_H_ */
