/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Fortified version of standard functions, used when _FORTIFY_SOURCE is enabled. We use compiler
 * builtins (e.g. __builtin___memcpy_chk), which, depending on their argument, call either the
 * normal version (memcpy) or the version with runtime check (__memcpy_chk).
 *
 * For more information, see:
 * - 'man feature_test_macros' and _FORTIFY_SOURCE
 * - https://gcc.gnu.org/onlinedocs/gcc/Object-Size-Checking.html
 * - glibc sources: string_fortified.h, stdio2.h
 *
 * This file will conflict with regular libc headers (e.g. <stdio.h>, <string.h>). That should not
 * be an issue because in no-stdlib context you should not use most libc headers, and in stdlib
 * context you should not use "api.h".
 *
 * However, if for any reason you need to include both "api.h" and headers such as <stdio.h>
 * alongside each other, you can disable this file (and prevent the conflict) by defining
 * USE_STDLIB:
 *
 *     #include <stdio.h>
 *     #include <string.h>
 *
 *     #define USE_STDLIB
 *     #include "api.h"
 */

#ifndef _API_FORTIFIED_H_
#define _API_FORTIFIED_H_

#ifndef API_H
# error This file should only be included inside api.h.
#endif

#define __api_bos0(ptr) __builtin_object_size(ptr, 0)
#define __api_bos(ptr) __builtin_object_size(ptr, __USE_FORTIFY_LEVEL > 1)

#define memcpy(dest, src, count) \
    __builtin___memcpy_chk(dest, src, count, __api_bos0(dest))

#define memmove(dest, src, count) \
    __builtin___memmove_chk(dest, src, count, __api_bos0(dest))

#define memset(dest, ch, count) \
    __builtin___memset_chk(dest, ch, count, __api_bos0(dest))

#define vsnprintf(buf, buf_size, fmt, ap) \
    __builtin___vsnprintf_chk(buf, buf_size, __USE_FORTIFY_LEVEL - 1, __api_bos(buf), fmt, ap)

#define snprintf(buf, buf_size, fmt...) \
    __builtin___snprintf_chk(buf, buf_size, __USE_FORTIFY_LEVEL - 1, __api_bos(buf), fmt)


#endif /* _API_FORTIFIED_H_ */
