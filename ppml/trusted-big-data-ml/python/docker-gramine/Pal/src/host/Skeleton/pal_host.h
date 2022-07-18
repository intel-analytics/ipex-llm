/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains definition of PAL host ABI.
 */

#ifndef PAL_HOST_H
#define PAL_HOST_H

#ifndef IN_PAL
#error "cannot be included outside PAL"
#endif

typedef struct {
    /* TSAI: Here we define the internal types of PAL_HANDLE in PAL design, user has not to access
     * the content inside the handle, also there is no need to allocate the internal handles, so we
     * hide the type name of these handles on purpose.
     */
    PAL_HDR hdr;
    uint32_t flags;

    /*
     * PAL has different kinds of handles, so you probably want to add an union with data specific
     * for following types:
     * - file,
     * - pipe,
     * - eventfd,
     * - dev,
     * - dir,
     * - sock,
     * - process,
     * - thread,
     * - event.
     * Note that this is just a hint, not a requirement. You can check the Linux PAL for a sample
     * implementation.
     */
}* PAL_HANDLE;

#endif /* PAL_HOST_H */
