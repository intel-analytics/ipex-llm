/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include "api.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_table.h"
#include "shim_thread.h"
#include "shim_types.h"

long shim_do_getuid(void) {
    struct shim_thread* current = get_cur_thread();
    lock(&current->lock);
    uid_t uid = current->uid;
    unlock(&current->lock);
    return uid;
}

long shim_do_getgid(void) {
    struct shim_thread* current = get_cur_thread();
    lock(&current->lock);
    gid_t gid = current->gid;
    unlock(&current->lock);
    return gid;
}

long shim_do_geteuid(void) {
    struct shim_thread* current = get_cur_thread();
    lock(&current->lock);
    uid_t euid = current->euid;
    unlock(&current->lock);
    return euid;
}

long shim_do_getegid(void) {
    struct shim_thread* current = get_cur_thread();
    lock(&current->lock);
    gid_t egid = current->egid;
    unlock(&current->lock);
    return egid;
}

long shim_do_setuid(uid_t uid) {
    struct shim_thread* current = get_cur_thread();
    lock(&current->lock);
    current->euid = uid;
    unlock(&current->lock);
    return 0;
}

long shim_do_setgid(gid_t gid) {
    struct shim_thread* current = get_cur_thread();
    lock(&current->lock);
    current->egid = gid;
    unlock(&current->lock);
    return 0;
}

#define NGROUPS_MAX 65536 /* # of supplemental group IDs; has to be same as host OS */

long shim_do_setgroups(int gidsetsize, gid_t* grouplist) {
    if (gidsetsize < 0 || (unsigned int)gidsetsize > NGROUPS_MAX)
        return -EINVAL;

    struct shim_thread* current = get_cur_thread();
    if (gidsetsize == 0) {
        free(current->groups_info.groups);
        current->groups_info.groups = NULL;
        current->groups_info.count = 0;
        return 0;
    }

    if (!is_user_memory_readable(grouplist, gidsetsize * sizeof(gid_t)))
        return -EFAULT;

    size_t groups_len = (size_t)gidsetsize;
    gid_t* groups = (gid_t*)malloc(groups_len * sizeof(*groups));
    if (!groups) {
        return -ENOMEM;
    }
    for (size_t i = 0; i < groups_len; i++) {
        groups[i] = grouplist[i];
    }

    void* old_groups = NULL;
    current->groups_info.count = groups_len;
    old_groups = current->groups_info.groups;
    current->groups_info.groups = groups;

    free(old_groups);

    return 0;
}

long shim_do_getgroups(int gidsetsize, gid_t* grouplist) {
    if (gidsetsize < 0)
        return -EINVAL;

    if (!is_user_memory_writable(grouplist, gidsetsize * sizeof(gid_t)))
        return -EFAULT;

    struct shim_thread* current = get_cur_thread();
    size_t ret_size = current->groups_info.count;

    if (gidsetsize) {
        if (ret_size > (size_t)gidsetsize) {
            return -EINVAL;
        }

        for (size_t i = 0; i < ret_size; i++) {
            grouplist[i] = current->groups_info.groups[i];
        }
    }

    return (int)ret_size;
}
