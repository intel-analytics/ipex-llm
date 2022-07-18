/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation */

#ifndef COMMON_LOG_H
#define COMMON_LOG_H

#include "callbacks.h"

enum {
    LOG_LEVEL_NONE    = 0,
    LOG_LEVEL_ERROR   = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_DEBUG   = 3,
    LOG_LEVEL_TRACE   = 4,
    LOG_LEVEL_ALL     = 5,
};

/* All of them implicitly append a newline at the end of the message. */
#define log_always(fmt...)   _log(LOG_LEVEL_NONE, fmt)
#define log_error(fmt...)    _log(LOG_LEVEL_ERROR, fmt)
#define log_warning(fmt...)  _log(LOG_LEVEL_WARNING, fmt)
#define log_debug(fmt...)    _log(LOG_LEVEL_DEBUG, fmt)
#define log_trace(fmt...)    _log(LOG_LEVEL_TRACE, fmt)

#endif /* COMMON_LOG_H */
