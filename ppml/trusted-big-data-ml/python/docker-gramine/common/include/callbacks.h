/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation */

/*
 * Defines a set of callbacks that the common library expects from environment it is linked into.
 * Currently, the common library expects `shim_`-prefixed callbacks from LibOS, `pal_`-prefixed
 * callbacks from PAL, and not-prefixed callbacks from all other environments (e.g., PAL regression
 * tests and non-Gramine programs). This header aliases the actual callback implementations, i.e.,
 * `shim_abort()` is aliased as `abort()` for use by the common library.
 *
 * Strictly speaking, current Gramine doesn't need different callback names for LibOS, PAL and
 * other environments. We introduce this notation for the future change where LibOS and PAL will be
 * statically linked together in a single binary (thus, we want to avoid name collisions in
 * callbacks).
 *
 * All environments should implement `_log` and `_abort` callbacks, and can optionally implement
 * `describe_location`.
 */

#ifndef COMMON_CALLBACKS_H
#define COMMON_CALLBACKS_H

#include <stddef.h>
#include <stdint.h>
#include <stdnoreturn.h>

#ifdef IN_SHIM

#define _log shim_log
#define abort shim_abort
#define describe_location shim_describe_location

#elif IN_PAL

#define _log pal_log
#define abort pal_abort
#define describe_location pal_describe_location

#endif

/* Recommended buffer size for `describe_location`. */
#define LOCATION_BUF_SIZE 128

/*
 * Output a formatted log message at a specific level (or drop it, if configured to do so).
 *
 * Used by `log_*` macros in `log.h`.
 */
void _log(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

/*
 * Terminate the process. Should perform an equivalent of `_exit(ENOTRECOVERABLE)`.
 *
 * Used by assertions, sanitizers, etc.
 */
noreturn void abort(void);

/*
 * Describe the code under given address: function, source line, etc.
 *
 * Currently used in AddressSanitizer (`asan.c`) and PAL/LibOS crash handlers.
 *
 * This callback is optional to implement (the common library contains a default implementation,
 * defined as a weak symbol).
 */
void describe_location(uintptr_t addr, char* buf, size_t buf_size);

/* This is the default implementation of `describe_location`, and returns only the raw value
 * ("0x1234"). Your implementation might call it when it fails to determine more information. */
void default_describe_location(uintptr_t addr, char* buf, size_t buf_size);

#endif /* COMMON_CALLBACKS_H */
