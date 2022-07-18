/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2017 University of North Carolina at Chapel Hill and Fortanix, Inc. */

/*
 * This file defines the list API for the PAL and Library OS.
 */

#ifndef LIST_H
#define LIST_H

// Use a new list implementation

/* This list implementation stores a pointer to the next object and casts to
 * the object, rather than using offsetof().  We try to encapsulate this
 * change in a macro for declarations, which generates a type declaration for
 * each list object (giving marginally more help from the compiler
 * in detecting bugs.
 *
 * In particular, there is a small trade-off in that the association between
 * list heads and nodes is more explicit and a few more casting errors can be
 * caught by the compiler, but we add a parameter to some functions (well,
 * macros) to pass the field of the struct.
 */

/* How-to:
 *
 * Each list has a pointer (listp) type, and a node (list)type.  We assume
 * list nodes are embedded in a larger structure; the name of this structure
 * is used as part of the list type.
 *
 * To define a listp/list pair for a struct foo:
 *
 * DEFINE_LIST(foo);
 * struct foo {
 *   int x;
 *   LIST_TYPE(foo) list; // The list node
 * };
 *
 * DEFINE_LISTP(foo);
 * static LISTP_TYPE(foo) the_list = LISTP_INIT;
 *
 * -----
 *
 * From here, you can use LISTP_ADD variants to add an object from the list:
 *
 * struct foo *f = malloc(sizeof(struct foo));
 * f->x = 1;
 * INIT_LIST_HEAD(f, list); // The second parameter is the structure member
 * LISTP_ADD(f, &the_list, list);
 *
 * -----
 *
 * There are a number of add variants, some that add in a given position,
 * others that add to the head or the tail.
 *
 * You can search for an object using a variant of listp_for_each_entry. The
 * safe variants are safe against deletion.
 *
 * You can remove an object from a list using LISTP_DEL.
 *
 * In this example, we delete everything with a key bigger than 5.
 *
 * LIST_TYPE(foo) *f, *n; // n is not used, just for scratch space
 * LISTP_FOR_EACH_ENTRY_SAFE(f, n, &the_list, list) {
 *    if (f->x > 4) {
 *         LISTP_DEL(f, &the_list, list);
 *         free(f);
 *    }
 * }
 *
 *
 * LISTP_SPLICE moves an entire listp onto another, and list_move_tail takes
 * an element off of one list and places it on another.
 *
 * static LISTP_TYPE(foo) other_list; // Assume it is full of goodies
 *  // Move everything on other_list to the_list
 * LISTP_SPLICE_TAIL(&other_list, &the_list, list, foo); // the third argument
 *                                                       // is the field; the
 *                                                       // fourth is the type
 *                                                       // of the nodes (not
 *                                                       // the head pointer).
 *
 * // Use LISTP_EMPTY to test for emptiness of the list
 * assert(LISTP_EMPTY(&other_ist));
 *
 *  // Now move back anythign less than 6 back to other_list
 * LISTP_FOR_EACH_ENTRY_SAFE(f, n, &the_list, list) {
 *    if (f->x < 6)
 *         LISTP_MOVE_TAIL(f, &other_list, &the_list, list);
 * }
 *
 */

// Maybe TODO?
//
// Change the order of (node, head, field) -> (head, node, field)
// drop the listp type to reduce code changes?
// Cleaner way to express types
// Add assertion to delete (in debugging mode) that item is on list
// There are a few places where knowing the listp for deletion is cumbersome;
//    maybe drop this requirement?

#include <stdbool.h>

#ifdef DEBUG
#include <assert.h>
#define LIST_ASSERT(COND) assert(COND)
#else
#define LIST_ASSERT(COND)
#endif

/* This is intentionally crafted to be in a canonical form on x64, so that it's not stripped from
 * siginfo_t::si_addr. It's also never accessible from usermode on Linux x86/x64, so it will always
 * raise an exception. */
#define INVALID_PTR_CANARY ((void*)0xffffffccccccccccUL)

#define LIST_TYPE(STRUCT_NAME)  struct list_head##_##STRUCT_NAME
#define LISTP_TYPE(STRUCT_NAME) struct listp##_##STRUCT_NAME

/* Declare the enclosing struct for convenience, on
 * the assumption that this is primarily used in structure
 * definitions, and harmless if duplicated. */
#define DEFINE_LIST(STRUCT_NAME)  \
    struct STRUCT_NAME;           \
    LIST_TYPE(STRUCT_NAME) {      \
        struct STRUCT_NAME* next; \
        struct STRUCT_NAME* prev; \
    }

/* We use LISTP for pointers to a list.  This project only really needs
 * doubly-linked lists.  We used hlists to get a single pointer for more
 * efficient hash tables, but they were still effectively doubly-linked
 * lists. */
#define DEFINE_LISTP(STRUCT)  \
    LISTP_TYPE(STRUCT) {      \
        struct STRUCT* first; \
    }

#define LISTP_INIT {NULL}

/* A node not on a list uses NULL; on a list, you
 * store self pointers */
#define INIT_LIST_HEAD(OBJECT, FIELD) \
    do {                              \
        (OBJECT)->FIELD.next = NULL;  \
        (OBJECT)->FIELD.prev = NULL;  \
    } while (0)

#define INIT_LISTP(OBJECT)      \
    do {                        \
        (OBJECT)->first = NULL; \
    } while (0)

#define LISTP_EMPTY(HEAD) ((HEAD)->first == NULL)

#define LIST_EMPTY(NODE, FIELD) ((NODE)->FIELD.next == NULL)

/* This helper takes 3 arguments - all should be containing structures,
 * and the field to use for the offset to the list node */
#define __LIST_ADD(NEW, NEXT, PREV, FIELD)       \
    do {                                         \
        __typeof__(NEW) __tmp_next = (NEXT);     \
        __typeof__(NEW) __tmp_prev = (PREV);     \
        __tmp_prev->FIELD.next     = (NEW);      \
        __tmp_next->FIELD.prev     = (NEW);      \
        (NEW)->FIELD.next          = __tmp_next; \
        (NEW)->FIELD.prev          = __tmp_prev; \
    } while (0)

#define LIST_ADD(NEW, HEAD, FIELD) __LIST_ADD(NEW, (HEAD)->FIELD.next, HEAD, FIELD)

#define LISTP_ADD(NEW, HEAD, FIELD)                                           \
    do {                                                                      \
        if ((HEAD)->first == NULL) {                                          \
            (HEAD)->first     = (NEW);                                        \
            (NEW)->FIELD.next = (NEW);                                        \
            (NEW)->FIELD.prev = (NEW);                                        \
        } else {                                                              \
            __LIST_ADD(NEW, (HEAD)->first, (HEAD)->first->FIELD.prev, FIELD); \
            (HEAD)->first = (NEW);                                            \
        }                                                                     \
    } while (0)

/* If NODE is defined, add NEW after NODE; if not,
 * put NEW at the front of the list */
#define LISTP_ADD_AFTER(NEW, NODE, HEAD, FIELD) \
    do {                                        \
        if (NODE)                               \
            LIST_ADD(NEW, NODE, FIELD);         \
        else                                    \
            LISTP_ADD(NEW, HEAD, FIELD);        \
    } while (0)

#define LIST_ADD_TAIL(NEW, HEAD, FIELD) __LIST_ADD(NEW, HEAD, (HEAD)->FIELD.prev, FIELD)

#define LISTP_ADD_TAIL(NEW, HEAD, FIELD)              \
    do {                                              \
        if ((HEAD)->first == NULL) {                  \
            (HEAD)->first     = (NEW);                \
            (NEW)->FIELD.next = (NEW);                \
            (NEW)->FIELD.prev = (NEW);                \
        } else                                        \
            LIST_ADD_TAIL(NEW, (HEAD)->first, FIELD); \
    } while (0)

#ifdef DEBUG
#define LIST_DEBUG_ERASE_NODE(NODE, FIELD)       \
    do {                                         \
        (NODE)->FIELD.prev = INVALID_PTR_CANARY; \
        (NODE)->FIELD.next = INVALID_PTR_CANARY; \
    } while (0)
#else
#define LIST_DEBUG_ERASE_NODE(NODE, FIELD) do {} while (0)
#endif

/* Or deletion needs to know the list root */
#define LISTP_DEL(NODE, HEAD, FIELD)                           \
    do {                                                       \
        if ((HEAD)->first == (NODE)) {                         \
            if ((NODE)->FIELD.next == (NODE)) {                \
                (HEAD)->first = NULL;                          \
            } else {                                           \
                (HEAD)->first = (NODE)->FIELD.next;            \
            }                                                  \
        }                                                      \
        LIST_ASSERT((NODE)->FIELD.prev->FIELD.next == (NODE)); \
        LIST_ASSERT((NODE)->FIELD.next->FIELD.prev == (NODE)); \
        (NODE)->FIELD.prev->FIELD.next = (NODE)->FIELD.next;   \
        (NODE)->FIELD.next->FIELD.prev = (NODE)->FIELD.prev;   \
        LIST_DEBUG_ERASE_NODE(NODE, FIELD);                    \
    } while (0)

#define LISTP_DEL_INIT(NODE, HEAD, FIELD) \
    do {                                  \
        LISTP_DEL(NODE, HEAD, FIELD);     \
        INIT_LIST_HEAD(NODE, FIELD);      \
    } while (0)

/* Keep vestigial TYPE and FIELD parameters to minimize disruption
 * when switching from Linux list implementation */
#define LISTP_FIRST_ENTRY(LISTP, TYPE, FIELD) ((LISTP)->first)

/* New API: return last entry in list */
#define LISTP_LAST_ENTRY(LISTP, TYPE, FIELD) ((LISTP)->first->FIELD.prev)

/* New API: return next entry in list */
#define LISTP_NEXT_ENTRY(NODE, LISTP, FIELD) \
    ((NODE) == (LISTP)->first->FIELD.prev ? NULL : (NODE)->FIELD.next)

/* New API: return previous entry in list */
#define LISTP_PREV_ENTRY(NODE, LISTP, FIELD) ((NODE) == (LISTP)->first ? NULL : (NODE)->FIELD.prev)

/* Vestigial - for compat with Linux list code; rename to listp?
 */
#define LIST_ENTRY(LISTP, TYPE, FIELD) (LISTP)

#define LISTP_FOR_EACH_ENTRY(CURSOR, HEAD, FIELD)                       \
    for (bool first_iter = ((CURSOR) = (HEAD)->first, !!(HEAD)->first); \
         first_iter || (CURSOR) != (HEAD)->first;                       \
         (CURSOR) = (CURSOR)->FIELD.next, first_iter = false)

#define LISTP_FOR_EACH_ENTRY_REVERSE(CURSOR, HEAD, FIELD)                             \
    for (bool first_iter =                                                            \
             ((CURSOR) = ((HEAD)->first ? (HEAD)->first->FIELD.prev : (HEAD)->first), \
             !!(HEAD)->first);                                                        \
         first_iter || ((CURSOR) && (CURSOR)->FIELD.next != (HEAD)->first);           \
         (CURSOR) = (CURSOR)->FIELD.prev, first_iter = false)

#define LISTP_FOR_EACH_ENTRY_SAFE(CURSOR, TMP, HEAD, FIELD)                                        \
    for (bool first_iter = ((CURSOR) = (HEAD)->first,                                              \
                           (TMP) = ((CURSOR) ? (CURSOR)->FIELD.next : (CURSOR)), !!(HEAD)->first); \
         (HEAD)->first &&                                                                          \
         (first_iter || (CURSOR) != (HEAD)->first);                                                \
         /* Handle the case where the first element was removed. */                                \
         first_iter = first_iter && (TMP) != (CURSOR) && (HEAD)->first == (TMP),                   \
             ((TMP) != (CURSOR) && ((CURSOR) = (TMP), (TMP) = (TMP)->FIELD.next)))

/* Continue safe iteration with CURSOR->next */
#define LISTP_FOR_EACH_ENTRY_SAFE_CONTINUE(CURSOR, TMP, HEAD, FIELD)    \
    for ((CURSOR) = (CURSOR)->FIELD.next, (TMP) = (CURSOR)->FIELD.next; \
         (CURSOR) != (HEAD)->first && (HEAD)->first; (CURSOR) = (TMP), (TMP) = (TMP)->FIELD.next)

/* Assertion code written in Gramine project */
#define CHECK_LIST_HEAD(TYPE, HEAD, FIELD)                               \
    do {                                                                 \
        TYPE pos;                                                        \
        LISTP_FOR_EACH_ENTRY(pos, HEAD, FIELD) {                         \
            assert((pos->FIELD.prev != pos && pos->FIELD.next != pos) || \
                   (pos->FIELD.prev == pos && pos->FIELD.next == pos));  \
            assert(pos->FIELD.prev->FIELD.next == pos);                  \
            assert(pos->FIELD.next->FIELD.prev == pos);                  \
        }                                                                \
    } while (0)

// Add NEW to OLD at position first (assuming first is all we need for now)
// Can probably drop TYPE with some preprocessor smarts
#define LISTP_SPLICE(NEW, OLD, FIELD, TYPE)                                      \
    do {                                                                         \
        if (!LISTP_EMPTY(NEW)) {                                                 \
            if (LISTP_EMPTY(OLD)) {                                              \
                (OLD)->first = (NEW)->first;                                     \
            } else {                                                             \
                struct TYPE* last_old                = (OLD)->first->FIELD.prev; \
                (OLD)->first->FIELD.prev->FIELD.next = (NEW)->first;             \
                (OLD)->first->FIELD.prev             = (NEW)->first->FIELD.prev; \
                (NEW)->first->FIELD.prev->FIELD.next = (OLD)->first;             \
                (NEW)->first->FIELD.prev             = last_old;                 \
                (OLD)->first                         = (NEW)->first;             \
            }                                                                    \
        }                                                                        \
    } while (0)

// Add NEW to OLD at last position
// Can probably drop TYPE with some preprocessor smarts
#define LISTP_SPLICE_TAIL(NEW, OLD, FIELD, TYPE)                                 \
    do {                                                                         \
        if (!LISTP_EMPTY(NEW)) {                                                 \
            if (LISTP_EMPTY(OLD)) {                                              \
                (OLD)->first = (NEW)->first;                                     \
            } else {                                                             \
                struct TYPE* last_old                = (OLD)->first->FIELD.prev; \
                last_old->FIELD.next                 = (NEW)->first;             \
                (OLD)->first->FIELD.prev             = (NEW)->first->FIELD.prev; \
                (NEW)->first->FIELD.prev->FIELD.next = (OLD)->first;             \
                (NEW)->first->FIELD.prev             = last_old;                 \
            }                                                                    \
        }                                                                        \
    } while (0)

#define LISTP_SPLICE_INIT(NEW, OLD, FIELD, TYPE) \
    do {                                         \
        LISTP_SPLICE(NEW, OLD, FIELD, TYPE);     \
        INIT_LISTP(NEW);                         \
    } while (0);

#define LISTP_SPLICE_TAIL_INIT(NEW, OLD, FIELD, TYPE) \
    do {                                              \
        LISTP_SPLICE_TAIL(NEW, OLD, FIELD, TYPE);     \
        INIT_LISTP(NEW);                              \
    } while (0);

// list_move_tail - delete from OLD, make tail of NEW
#define LISTP_MOVE_TAIL(NODE, NEW, OLD, FIELD) \
    do {                                       \
        LISTP_DEL_INIT(NODE, OLD, FIELD);      \
        LISTP_ADD_TAIL(NODE, NEW, FIELD);      \
    } while (0)

#endif  // LIST_H
