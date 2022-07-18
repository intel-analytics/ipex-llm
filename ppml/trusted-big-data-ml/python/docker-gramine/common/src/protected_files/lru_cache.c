/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 * Copyright (C) 2020 Intel Corporation
 */

/* TODO: add regression tests for this */

#include "assert.h"
#include "lru_cache.h"
#include "list.h"

#ifdef IN_TOOLS

#include <stdio.h>
#include <stdlib.h>

#define uthash_fatal(msg)                            \
    do {                                             \
        fprintf(stderr, "uthash error: %s\n", msg);  \
        exit(-1);                                    \
    } while(0)

#else

#include "api.h"

#define uthash_fatal(msg)                            \
    do {                                             \
        log_error("uthash error: %s", msg);          \
        abort();                                     \
    } while(0)

#endif

#include "uthash.h"

DEFINE_LIST(_lruc_list_node);
typedef struct _lruc_list_node {
    LIST_TYPE(_lruc_list_node) list;
    uint64_t key;
} lruc_list_node_t;
DEFINE_LISTP(_lruc_list_node);

typedef struct _lruc_map_node {
    uint64_t key;
    void* data;
    lruc_list_node_t* list_ptr;
    UT_hash_handle hh;
} lruc_map_node_t;

struct lruc_context {
    /* list and map both contain the same objects (list contains keys, map contains actual data).
     * They're kept in sync so that map is used for fast lookups and list is used for fast LRU.
     */
    LISTP_TYPE(_lruc_list_node) list;
    lruc_map_node_t* map;
    lruc_list_node_t* current; /* current head of the cache */
};

lruc_context_t* lruc_create(void) {
    lruc_context_t* lruc = calloc(1, sizeof(*lruc));
    if (!lruc)
        return NULL;

    INIT_LISTP(&lruc->list);
    lruc->map     = NULL;
    lruc->current = NULL;
    return lruc;
}

static lruc_map_node_t* get_map_node(lruc_context_t* lruc, uint64_t key) {
    lruc_map_node_t* mn = NULL;
    HASH_FIND(hh, lruc->map, &key, sizeof(key), mn);
    return mn;
}

void lruc_destroy(lruc_context_t* lruc) {
    struct _lruc_list_node* ln;
    struct _lruc_list_node* tmp;
    lruc_map_node_t* mn;

    LISTP_FOR_EACH_ENTRY_SAFE(ln, tmp, &lruc->list, list) {
        mn = get_map_node(lruc, ln->key);
        if (mn) {
            HASH_DEL(lruc->map, mn);
            free(mn);
        }
        LISTP_DEL(ln, &lruc->list, list);
        free(ln);
    }

    assert(LISTP_EMPTY(&lruc->list));
    assert(HASH_COUNT(lruc->map) == 0);
    free(lruc);
}

bool lruc_add(lruc_context_t* lruc, uint64_t key, void* data) {
    if (get_map_node(lruc, key))
        return false;

    lruc_map_node_t* map_node = calloc(1, sizeof(*map_node));
    if (!map_node)
        return false;

    lruc_list_node_t* list_node = calloc(1, sizeof(*list_node));
    if (!list_node) {
        free(map_node);
        return false;
    }

    list_node->key = key;
    map_node->key = key;
    LISTP_ADD(list_node, &lruc->list, list);
    map_node->data     = data;
    map_node->list_ptr = list_node;
    HASH_ADD(hh, lruc->map, key, sizeof(key), map_node);
    return true;
}

void* lruc_find(lruc_context_t* lruc, uint64_t key) {
    lruc_map_node_t* mn = get_map_node(lruc, key);
    if (mn)
        return mn->data;
    return NULL;
}

void* lruc_get(lruc_context_t* lruc, uint64_t key) {
    lruc_map_node_t* mn = get_map_node(lruc, key);
    if (!mn)
        return NULL;
    lruc_list_node_t* ln = mn->list_ptr;
    assert(ln != NULL);
    // move node to the front of the list
    LISTP_DEL(ln, &lruc->list, list);
    LISTP_ADD(ln, &lruc->list, list);
    return mn->data;
}

size_t lruc_size(lruc_context_t* lruc) {
    return HASH_COUNT(lruc->map);
}

void* lruc_get_first(lruc_context_t* lruc) {
    if (LISTP_EMPTY(&lruc->list))
        return NULL;

    lruc->current = LISTP_FIRST_ENTRY(&lruc->list, /*unused*/ 0, list);
    lruc_map_node_t* mn = get_map_node(lruc, lruc->current->key);
    assert(mn != NULL);
    return mn ? mn->data : NULL;
}

void* lruc_get_next(lruc_context_t* lruc) {
    if (LISTP_EMPTY(&lruc->list) || !lruc->current)
        return NULL;

    lruc->current = LISTP_NEXT_ENTRY(lruc->current, &lruc->list, list);
    if (!lruc->current)
        return NULL;

    lruc_map_node_t* mn = get_map_node(lruc, lruc->current->key);
    assert(mn != NULL);
    return mn ? mn->data : NULL;
}

void* lruc_get_last(lruc_context_t* lruc) {
    if (LISTP_EMPTY(&lruc->list))
        return NULL;

    lruc_list_node_t* ln = LISTP_LAST_ENTRY(&lruc->list, /*unused*/ 0, list);
    lruc_map_node_t* mn = get_map_node(lruc, ln->key);
    assert(mn != NULL);
    return mn ? mn->data : NULL;
}

void lruc_remove_last(lruc_context_t* lruc) {
    if (LISTP_EMPTY(&lruc->list))
        return;

    lruc_list_node_t* ln = LISTP_LAST_ENTRY(&lruc->list, /*unused*/ 0, list);
    LISTP_DEL(ln, &lruc->list, list);
    lruc_map_node_t* mn = get_map_node(lruc, ln->key);
    assert(mn != NULL);
    if (mn)
        HASH_DEL(lruc->map, mn);
    free(ln);
    free(mn);
}
