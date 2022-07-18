/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs for closing or polling PAL handles.
 */

#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

int _DkObjectClose(PAL_HANDLE object_handle) {
    const struct handle_ops* ops = HANDLE_OPS(object_handle);
    if (!ops)
        return -PAL_ERROR_BADHANDLE;

    int ret = 0;

    /* if the operation 'close' is defined, call the function. */
    if (ops->close)
        ret = ops->close(object_handle);

    /*
     * Chia-Che 12/7/2017:
     *   _DkObjectClose will free the object, unless the handle has a 'close' operation, and the
     *   operation returns a non-zero value (e.g., 1 for skipping free() or -ERRNO).
     */
    if (!ret)
        free(object_handle);

    return ret;
}

/*
 * TODO: whole LibOS assumes this never returns errors. We need to either make `_DkObjectClose`
 * never fail (from a quick look at the code, seems like it cannot return errors in practice) or
 * make this return an `int` and handle errors in all call sites (hard to do; in most places we
 * cannot handle them in a meaningful way).
 */
/* PAL call DkObjectClose: Close the given object handle. */
void DkObjectClose(PAL_HANDLE object_handle) {
    assert(object_handle);

    _DkObjectClose(object_handle);
}

int DkStreamsWaitEvents(size_t count, PAL_HANDLE* handle_array, pal_wait_flags_t* events,
                        pal_wait_flags_t* ret_events, uint64_t* timeout_us) {
    for (size_t i = 0; i < count; i++) {
        assert(handle_array[i]);
        assert(!UNKNOWN_HANDLE(handle_array[i]));
    }

    return _DkStreamsWaitEvents(count, handle_array, events, ret_events, timeout_us);
}
