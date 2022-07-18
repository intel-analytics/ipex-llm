/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Outer PAL logging interface. This is initialized separately to inner PAL, but (once it's
 * initialized) should output at the level and to the file specified in manifest.
 */

#ifndef SGX_LOG_H_
#define SGX_LOG_H_

extern int g_urts_log_level;
extern int g_urts_log_fd;

int urts_log_init(const char* path);

// TODO(mkow): We should make it cross-object-inlinable, ideally by enabling LTO, less ideally by
// pasting it here and making `inline`, but our current linker scripts prevent both.
void pal_log(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

#endif /* SGX_LOG_H_ */
