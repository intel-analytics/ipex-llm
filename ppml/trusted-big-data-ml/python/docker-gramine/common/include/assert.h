/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation */

/*
 * Define a common interface for assertions that builds for both the PAL and libOS.
 */

#ifndef ASSERT_H
#define ASSERT_H

#include "callbacks.h"
#include "log.h"

#define static_assert _Static_assert

/* TODO(mkow): We should actually use the standard `NDEBUG`, but that would require changes in the
 * build system.
 */
#ifdef DEBUG
/* This `if` is weird intentionally - not to have parentheses around `expr` to catch `assert(x = y)`
 * errors. */
#define assert(expr)                                                              \
    ({                                                                            \
        if (expr) {} else {                                                       \
            log_always("assert failed " __FILE__ ":%d %s", __LINE__, #expr);      \
            abort();                                                              \
        }                                                                         \
        (void)0;                                                                  \
    })
#else
#define assert(expr) ((void)0)
#endif

#endif /* ASSERT_H */
