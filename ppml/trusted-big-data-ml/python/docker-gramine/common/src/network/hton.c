/* Copyright (C) 1993,97,2002 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, write to the Free
   Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307 USA.  */

#include <endian.h>

#include "api.h"

static uint16_t __bswap_16(uint16_t x) {
    return __builtin_bswap32(x) >> 16;
}

static uint32_t __bswap_32(uint32_t x) {
    return __builtin_bswap32(x);
}

uint32_t __htonl(uint32_t x) {
#if __BYTE_ORDER == __BIG_ENDIAN
    return x;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
    return __bswap_32(x);
#else
#error "What kind of system is this?"
#endif
}

uint32_t __ntohl(uint32_t x) {
    return __htonl(x);
}

uint16_t __htons(uint16_t x) {
#if __BYTE_ORDER == __BIG_ENDIAN
    return x;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
    return __bswap_16(x);
#else
#error "What kind of system is this?"
#endif
}

uint16_t __ntohs(uint16_t x) {
    return __htons(x);
}
