/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs to set up handlers of exceptions issued by the host, and the methods to
 * pass the exceptions to the upcalls.
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

__attribute__((unused)) // Should be used by exception handling logic
static pal_event_handler_t g_handlers[PAL_EVENT_NUM_BOUND] = {0};
