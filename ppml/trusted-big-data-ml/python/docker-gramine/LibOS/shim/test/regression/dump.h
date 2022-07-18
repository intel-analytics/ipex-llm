/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#ifndef DUMP_H_
#define DUMP_H_

/*
 * Dumps the whole directory tree, displaying link targets and file contents. In addition to
 * examining the output, can be used to verify that:
 *
 * - all files can be examined with `lstat`,
 * - directories can be listed,
 * - symlinks can be examined with `readlink`,
 * - regular files can be opened and read.
 *
 * Returns 0 on success, -1 on failure.
 *
 * Example output:
 *
 *     dir: directory
 *     dir/file: file
 *     [dir/file] contents of dir/file
 *     dir/dir2: directory
 *     dir/dir2/link: link: /link/target
 */
int dump_path(const char* path);

#endif /* DUMP_H_ */
