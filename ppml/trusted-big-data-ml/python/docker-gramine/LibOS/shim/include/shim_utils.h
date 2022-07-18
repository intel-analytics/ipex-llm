/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

#ifndef _SHIM_UTILS_H_
#define _SHIM_UTILS_H_

#include "api.h"
#include "pal.h"
#include "shim_handle.h"
#include "shim_types.h"
#include "toml.h"

struct shim_handle;

/* quick hash function based on Robert Jenkins' hash algorithm */
static inline uint64_t hash64(uint64_t key) {
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

/* heap allocation functions */
int init_slab(void);

void* malloc(size_t size);
void free(void* mem);
void* malloc_copy(const void* mem, size_t size);

/* ELF binary loading */
struct link_map;
int init_elf_objects(void);
int check_elf_object(struct shim_handle* file);
int load_elf_object(struct shim_handle* file, struct link_map** out_map);
int load_elf_interp(struct link_map* exec_map);
int load_and_check_exec(const char* path, const char** argv, struct shim_handle** out_exec,
                        char*** out_new_argv);
noreturn void execute_elf_object(struct link_map* exec_map, void* argp, elf_auxv_t* auxp);
void remove_loaded_elf_objects(void);
int init_brk_from_executable(struct link_map* exec_map);
int register_library(const char* name, unsigned long load_address);

/* gdb debugging support */
int init_r_debug(void);
void remove_r_debug(void* addr);
void append_r_debug(const char* uri, void* addr);
void clean_link_map_list(void);

/* Get the IPC pipe uri associated with `vmid`. */
int vmid_to_uri(IDTYPE vmid, char* uri, size_t uri_len);

/* create unique files/pipes */
int create_pipe(char* name, char* uri, size_t size, PAL_HANDLE* hdl, bool use_vmid_for_name);

/* Asynchronous event support */
int init_async_worker(void);
int64_t install_async_event(PAL_HANDLE object, unsigned long time,
                            void (*callback)(IDTYPE caller, void* arg), void* arg);
struct shim_thread* terminate_async_worker(void);

extern const toml_table_t* g_manifest_root;

int read_exact(PAL_HANDLE handle, void* buf, size_t size);
int write_exact(PAL_HANDLE handle, void* buf, size_t size);

static inline uint64_t timespec_to_us(const struct __kernel_timespec* ts) {
    return ts->tv_sec * TIME_US_IN_S + ts->tv_nsec / TIME_NS_IN_US;
}

#endif /* _SHIM_UTILS_H */
