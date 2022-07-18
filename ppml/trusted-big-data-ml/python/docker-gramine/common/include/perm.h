/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Human-readable macros for common file permissions.
 * Inspired by Linux patch by Ingo Molnar (https://lwn.net/Articles/696231/).
 */

#ifndef PERM_H
#define PERM_H

#define PERM_r________  0400
#define PERM_r__r_____  0440
#define PERM_r__r__r__  0444

#define PERM_rw_______  0600
#define PERM_rw_r_____  0640
#define PERM_rw_r__r__  0644
#define PERM_rw_rw_r__  0664
#define PERM_rw_rw_rw_  0666

#define PERM_r_x______  0500
#define PERM_r_xr_x___  0550
#define PERM_r_xr_xr_x  0555

#define PERM_rwx______  0700
#define PERM_rwxr_x___  0750
#define PERM_rwxr_xr_x  0755
#define PERM_rwxrwxr_x  0775
#define PERM_rwxrwxrwx  0777

#endif /* PERM_H */
