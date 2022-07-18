/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * This file defines handlers for undefined behavior sanitization (UBSan).
 *
 * Normally, code compiled with UBSan is linked against a special library (libubsan). Unfortunately,
 * that library depends on libc, and it's not easy to adapt to a no-stdlib setting. Instead, we
 * create our own minimal handlers for UBSan errors.
 *
 * For more information, see:
 *
 * - UBSan documentation: https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
 *
 * - libubsan source code in LLVM repository: https://github.com/llvm/llvm-project/
 *   (compiler-rt/lib/ubsan/ubsan_handlers.cpp)
 */

#include <stdbool.h>
#include <stdint.h>

#include "api.h"
#include "log.h"

/* Type definitions (adapted from libubsan) */

struct type_descriptor;

struct source_location {
    const char* filename;
    uint32_t line;
    uint32_t column;
};

struct type_mismatch_data {
    struct source_location loc;
    const struct type_descriptor* type;
    uint8_t log_alignment;
    uint8_t type_check_kind;
};

typedef uintptr_t value_handle;

static void ubsan_log_location(struct source_location* loc) {
    log_error("ubsan: %s:%d:%d", loc->filename, loc->line, loc->column);
}

/*
 * Simple handlers: print source location and a format string based on parameters.
 *
 * Note that in UBSan API, the first parameter for some of these handlers is not a source_location,
 * but a bigger struct that begins with source_location (and contains additional details, which we
 * ignore).
 */

#define HANDLER(name)       __ubsan_handle_##name
#define HANDLER_ABORT(name) __ubsan_handle_##name##_abort

#define __UBSAN_SIMPLE_HANDLER(name, fmt, params, ...) \
    void HANDLER(name) params;                         \
    void HANDLER(name) params {                        \
        log_error("ubsan: " fmt, ##__VA_ARGS__);       \
        ubsan_log_location(loc);                       \
    }                                                  \
    void HANDLER_ABORT(name) params;                   \
    void HANDLER_ABORT(name) params {                  \
        HANDLER(name)(loc, ##__VA_ARGS__);             \
        abort();                                       \
    }

#define UBSAN_SIMPLE_HANDLER_0(name, fmt) \
    __UBSAN_SIMPLE_HANDLER(name, fmt, (struct source_location* loc))

#define UBSAN_SIMPLE_HANDLER_1(name, fmt) \
    __UBSAN_SIMPLE_HANDLER(name, fmt, (struct source_location* loc, value_handle a), a)

#define UBSAN_SIMPLE_HANDLER_2(name, fmt)                                           \
    __UBSAN_SIMPLE_HANDLER(name, fmt, (struct source_location* loc, value_handle a, \
                                       value_handle b),                             \
                           a, b)

#define UBSAN_SIMPLE_HANDLER_3(name, fmt)                                           \
    __UBSAN_SIMPLE_HANDLER(name, fmt, (struct source_location* loc, value_handle a, \
                                       value_handle b, value_handle c),             \
                           a, b, c)

UBSAN_SIMPLE_HANDLER_2(add_overflow,
                       "overflow: %ld + %ld")
UBSAN_SIMPLE_HANDLER_2(sub_overflow,
                       "overflow: %ld - %ld")
UBSAN_SIMPLE_HANDLER_2(mul_overflow,
                       "overflow: %ld * %ld")
UBSAN_SIMPLE_HANDLER_2(divrem_overflow,
                       "overflow: %ld / %ld")
UBSAN_SIMPLE_HANDLER_1(negate_overflow,
                       "overflow: - %ld")
UBSAN_SIMPLE_HANDLER_2(pointer_overflow,
                       "pointer overflow: applying offset to 0x%lx produced 0x%lx")
UBSAN_SIMPLE_HANDLER_1(load_invalid_value,
                       "load of invalid value for bool or enum: %ld")
UBSAN_SIMPLE_HANDLER_0(builtin_unreachable,
                       "__builtin_unreachable")
UBSAN_SIMPLE_HANDLER_2(shift_out_of_bounds,
                       "shift out of bounds: %ld by %ld")
UBSAN_SIMPLE_HANDLER_1(out_of_bounds,
                       "array index out of bounds: %ld")
UBSAN_SIMPLE_HANDLER_1(vla_bound_not_positive,
                       "variable-length array bound is not positive: %ld")
UBSAN_SIMPLE_HANDLER_1(float_cast_overflow,
                       "float cast overflow from 0x%lx")
UBSAN_SIMPLE_HANDLER_0(missing_return,
                       "execution reached end of value-returning function without returning a "
                       "value")
UBSAN_SIMPLE_HANDLER_2(implicit_conversion,
                       "implicit conversion changed the value %ld to %ld")
UBSAN_SIMPLE_HANDLER_1(type_mismatch,
                       "type mismatch for pointer 0x%lx")
UBSAN_SIMPLE_HANDLER_3(alignment_assumption,
                       "alignment assumption failed for pointer 0x%lx (%ld byte alignment, offset "
                       "%ld)")
UBSAN_SIMPLE_HANDLER_0(nonnull_arg,
                       "null pointer passed as an argument declared to never be null")
UBSAN_SIMPLE_HANDLER_0(nonnull_return_v1,
                       "null pointer returned from function declared to never return null")
UBSAN_SIMPLE_HANDLER_0(nullability_return_v1,
                       "null pointer returned from function declared to never return null")

/* More complex handlers, displaying additional information. */

void __ubsan_handle_type_mismatch_v1(struct type_mismatch_data* data, value_handle pointer);
void __ubsan_handle_type_mismatch_v1_abort(struct type_mismatch_data* data, value_handle pointer);

void __ubsan_handle_type_mismatch_v1(struct type_mismatch_data* data, value_handle pointer) {
    if (!pointer) {
        log_error("ubsan: null pointer dereference");
    } else if (pointer & (((uintptr_t)1 << data->log_alignment) - 1)) {
        log_error("ubsan: misaligned address %p for type with alignment %lu", (void*)pointer,
                  (uintptr_t)1 << data->log_alignment);
    } else {
        log_error("ubsan: address %p with insufficient space for object", (void*)pointer);
    }

    ubsan_log_location(&data->loc);
}

void __ubsan_handle_type_mismatch_v1_abort(struct type_mismatch_data* data, value_handle pointer) {
    __ubsan_handle_type_mismatch_v1(data, pointer);
    abort();
}
