#ifndef _ATOMIC_H_
#define _ATOMIC_H_

/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2017 Fortanix Inc, and University of North Carolina at Chapel Hill.
 *
 * This file defines atomic operations (And barriers) for use in
 * Gramine.
 *
 * The atomic operation assembly code is taken from musl libc, which
 * is subject to the MIT license.
 *
 * At this point, we primarily focus on x86_64; there are some vestigial
 * 32-bit definitions here, but a more portable version would need to
 * move and reimplement portions of this for 32-bit x86 (or other architectures).
 */

/*
/----------------------------------------------------------------------
Copyright (C) 2005-2014 Rich Felker, et al.

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    ----------------------------------------------------------------------
*/

#include <stdbool.h>
#include <stdint.h>

struct atomic_int {
    volatile int64_t counter;
};

#define ATOMIC_INIT(i)      { (i) }

#endif /* _ATOMIC_H_ */
