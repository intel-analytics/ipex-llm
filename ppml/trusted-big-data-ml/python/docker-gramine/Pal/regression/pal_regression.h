/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation */

#ifndef PAL_REGRESSION_H
#define PAL_REGRESSION_H

#include "pal.h"

void __attribute__((format(printf, 1, 2))) pal_printf(const char* fmt, ...);
void __attribute__((format(printf, 2, 3))) _log(int level, const char* fmt, ...);

#endif /* PAL_REGRESSION_H */
