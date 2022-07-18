/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains implementation of Drawbridge event synchronization APIs.
 */

#include "assert.h"
#include "pal_error.h"
#include "pal_internal.h"

int _DkEventCreate(PAL_HANDLE* handle_ptr, bool init_signaled, bool auto_clear) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

void _DkEventSet(PAL_HANDLE handle) {
    assert(0);
}

void _DkEventClear(PAL_HANDLE handle) {
    assert(0);
}

int _DkEventWait(PAL_HANDLE handle, uint64_t* timeout_us) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int event_close(PAL_HANDLE handle) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

struct handle_ops g_event_ops = {
    .close = event_close,
};
