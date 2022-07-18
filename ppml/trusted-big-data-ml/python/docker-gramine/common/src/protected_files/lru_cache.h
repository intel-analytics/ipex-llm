/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 * Copyright (C) 2020 Intel Corporation
 */

/* Least-recently used cache, used by the protected file implementation for optimizing
   data and MHT node access */

#ifndef LRU_CACHE_H_
#define LRU_CACHE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct lruc_context;
typedef struct lruc_context lruc_context_t;

lruc_context_t* lruc_create(void);
void lruc_destroy(lruc_context_t* context);
bool lruc_add(lruc_context_t* context, uint64_t key, void* data); // key must not already exist
void* lruc_get(lruc_context_t* context, uint64_t key);
void* lruc_find(lruc_context_t* context,
                uint64_t key); // only returns the object, does not bump it to the head
size_t lruc_size(lruc_context_t* context);
void* lruc_get_first(lruc_context_t* context);
void* lruc_get_next(lruc_context_t* context);
void* lruc_get_last(lruc_context_t* context);
void lruc_remove_last(lruc_context_t* context);

void lruc_test(void);

#endif /* LRU_CACHE_H_ */
