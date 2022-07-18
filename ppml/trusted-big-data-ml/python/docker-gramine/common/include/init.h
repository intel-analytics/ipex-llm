/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#ifndef INIT_H_
#define INIT_H_

/*
 * Call the constructors specified in `.init_array`. Should be called during initialization.
 *
 * NOTE: Glibc handles `.init_array` by itself, so normal executables compiled against Glibc (e.g.
 * Linux-SGX untrusted runtime) should not call this.
 */
void call_init_array(void);

#endif /* INIT_H_ */
