/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains the main function of the PAL loader, which loads and processes environment,
 * arguments and manifest.
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

/* must implement "pal_start", and call "pal_main" inside */
void pal_start(void);

void _DkGetAvailableUserAddressRange(void** out_start, void** out_end) {
    /* needs to be implemented */
}
