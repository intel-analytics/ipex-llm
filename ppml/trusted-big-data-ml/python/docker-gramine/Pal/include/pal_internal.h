/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains definitions of functions, variables and data structures for internal uses.
 */

#ifndef PAL_INTERNAL_H
#define PAL_INTERNAL_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>

#include "api.h"
#include "log.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_topology.h"
#include "toml.h"

#ifndef IN_PAL
#error "pal_internal.h can only be included in PAL"
#endif

/*
 * Part of PAL state which is common to all PALs (trusted part, in case of TEE PALs).
 * Most of it is actually very Linux-specific, we'll need to refactor it if we ever add a non-Linux
 * PAL.
 */
struct pal_common_state {
    uint64_t instance_id;
    PAL_HANDLE parent_process;
    const char* raw_manifest_data;
};
extern struct pal_common_state g_pal_common_state;
extern struct pal_public_state g_pal_public_state;

/* handle_ops is the operators provided for each handler type. They are mostly used by
 * stream-related PAL calls, but can also be used by some others in special ways. */
struct handle_ops {
    /* 'getrealpath' return the real path that represent the handle */
    const char* (*getrealpath)(PAL_HANDLE handle);

    /* 'getname' is used by DkStreamGetName. It's different from 'getrealpath' */
    int (*getname)(PAL_HANDLE handle, char* buffer, size_t count);

    /* 'open' is used by DkStreamOpen. 'handle' is a preallocated handle, 'type' will be a
     * normalized prefix, 'uri' is the remaining string of uri. access, share, create, and options
     * follow the same flags defined for DkStreamOpen in pal.h. */
    int (*open)(PAL_HANDLE* handle, const char* type, const char* uri, enum pal_access access,
                pal_share_flags_t share, enum pal_create_mode create, pal_stream_options_t options);

    /* 'read' and 'write' is used by DkStreamRead and DkStreamWrite, so they have exactly same
     * prototype as them. */
    int64_t (*read)(PAL_HANDLE handle, uint64_t offset, uint64_t count, void* buffer);
    int64_t (*write)(PAL_HANDLE handle, uint64_t offset, uint64_t count, const void* buffer);

    /* 'readbyaddr' and 'writebyaddr' are the same as read and write, but with extra field to
     * specify address */
    int64_t (*readbyaddr)(PAL_HANDLE handle, uint64_t offset, uint64_t count, void* buffer,
                          char* addr, size_t addrlen);
    int64_t (*writebyaddr)(PAL_HANDLE handle, uint64_t offset, uint64_t count, const void* buffer,
                           const char* addr, size_t addrlen);

    /* 'close' and 'delete' is used by DkObjectClose and DkStreamDelete, 'close' will close the
     * stream, while 'delete' actually destroy the stream, such as deleting a file or shutting
     * down a socket */
    int (*close)(PAL_HANDLE handle);
    int (*delete)(PAL_HANDLE handle, enum pal_delete_mode delete_mode);

    /*
     * 'map' and 'unmap' will map or unmap the handle into memory space, it's not necessary mapped
     * by mmap, so unmap also needs 'handle' to deal with special cases.
     *
     * Common PAL code will ensure that *address, offset, and size are page-aligned. 'address'
     * should not be NULL.
     */
    int (*map)(PAL_HANDLE handle, void** address, pal_prot_flags_t prot, uint64_t offset,
               uint64_t size);

    /* 'setlength' is used by DkStreamFlush. It truncate the stream to certain size. */
    int64_t (*setlength)(PAL_HANDLE handle, uint64_t length);

    /* 'flush' is used by DkStreamFlush. It syncs the stream to the device */
    int (*flush)(PAL_HANDLE handle);

    /* 'waitforclient' is used by DkStreamWaitforClient. It accepts an connection */
    int (*waitforclient)(PAL_HANDLE server, PAL_HANDLE* client, pal_stream_options_t options);

    /* 'attrquery' is used by DkStreamAttributesQuery. It queries the attributes of a stream */
    int (*attrquery)(const char* type, const char* uri, PAL_STREAM_ATTR* attr);

    /* 'attrquerybyhdl' is used by DkStreamAttributesQueryByHandle. It queries the attributes of
     * a stream handle */
    int (*attrquerybyhdl)(PAL_HANDLE handle, PAL_STREAM_ATTR* attr);

    /* 'attrsetbyhdl' is used by DkStreamAttributesSetByHandle. It queries the attributes of
     * a stream handle */
    int (*attrsetbyhdl)(PAL_HANDLE handle, PAL_STREAM_ATTR* attr);

    /* 'rename' is used to change name of a stream, or reset its share option */
    int (*rename)(PAL_HANDLE handle, const char* type, const char* uri);
};

extern const struct handle_ops* g_pal_handle_ops[];

static inline const struct handle_ops* HANDLE_OPS(PAL_HANDLE handle) {
    int _type = PAL_GET_TYPE(handle);
    if (_type < 0 || _type >= PAL_HANDLE_TYPE_BOUND)
        return NULL;
    return g_pal_handle_ops[_type];
}

/* We allow dynamic size handle allocation. Here is some macro to help deciding the actual size of
 * the handle */
extern PAL_HANDLE _h;
#define HANDLE_SIZE(type) (sizeof(*_h))

static inline size_t handle_size(PAL_HANDLE handle) {
    return sizeof(*handle);
}

/*
 * failure notify. The rountine is called whenever a PAL call return error code. As the current
 * design of PAL does not return error code directly, we rely on DkAsynchronousEventUpcall to handle
 * PAL call error. If the user does not set up a upcall, the error code will be ignored. Ignoring
 * PAL error code can be a possible optimization for SHIM.
 */
void notify_failure(unsigned long error);

int add_preloaded_range(uintptr_t start, uintptr_t end, const char* comment);

#define IS_ALLOC_ALIGNED(addr)     IS_ALIGNED_POW2(addr, g_pal_public_state.alloc_align)
#define IS_ALLOC_ALIGNED_PTR(addr) IS_ALIGNED_PTR_POW2(addr, g_pal_public_state.alloc_align)
#define ALLOC_ALIGN_UP(addr)       ALIGN_UP_POW2(addr, g_pal_public_state.alloc_align)
#define ALLOC_ALIGN_UP_PTR(addr)   ALIGN_UP_PTR_POW2(addr, g_pal_public_state.alloc_align)
#define ALLOC_ALIGN_DOWN(addr)     ALIGN_DOWN_POW2(addr, g_pal_public_state.alloc_align)
#define ALLOC_ALIGN_DOWN_PTR(addr) ALIGN_DOWN_PTR_POW2(addr, g_pal_public_state.alloc_align)

/*!
 * \brief Main initialization function.
 *
 * This function must be called by the host-specific loader.
 *
 * \param instance_id     Current instance ID.
 * \param exec_uri        Executable URI.
 * \param parent_process  Parent process if it's a child.
 * \param first_thread    First thread handle.
 * \param arguments       Application arguments.
 * \param environments    Environment variables.
 */
noreturn void pal_main(uint64_t instance_id, PAL_HANDLE parent_process, PAL_HANDLE first_thread,
                       const char** arguments, const char** environments);

/* For initialization */

void _DkGetAvailableUserAddressRange(void** out_start, void** out_end);
bool _DkCheckMemoryMappable(const void* addr, size_t size);
unsigned long _DkMemoryQuota(void);
unsigned long _DkMemoryAvailableQuota(void);
// Returns 0 on success, negative PAL code on failure
int _DkGetCPUInfo(struct pal_cpu_info* info);

/* Internal DK calls, in case any of the internal routines needs to use them */
/* DkStream calls */
int _DkStreamOpen(PAL_HANDLE* handle, const char* uri, enum pal_access access,
                  pal_share_flags_t share, enum pal_create_mode create,
                  pal_stream_options_t options);
int _DkStreamDelete(PAL_HANDLE handle, enum pal_delete_mode delete_mode);
int64_t _DkStreamRead(PAL_HANDLE handle, uint64_t offset, uint64_t count, void* buf, char* addr,
                      int addrlen);
int64_t _DkStreamWrite(PAL_HANDLE handle, uint64_t offset, uint64_t count, const void* buf,
                       const char* addr, int addrlen);
int _DkStreamAttributesQuery(const char* uri, PAL_STREAM_ATTR* attr);
int _DkStreamAttributesQueryByHandle(PAL_HANDLE hdl, PAL_STREAM_ATTR* attr);
int _DkStreamMap(PAL_HANDLE handle, void** addr_ptr, pal_prot_flags_t prot, uint64_t offset,
                 uint64_t size);
int _DkStreamUnmap(void* addr, uint64_t size);
int64_t _DkStreamSetLength(PAL_HANDLE handle, uint64_t length);
int _DkStreamFlush(PAL_HANDLE handle);
int _DkStreamGetName(PAL_HANDLE handle, char* buf, size_t size);
const char* _DkStreamRealpath(PAL_HANDLE hdl);
int _DkSendHandle(PAL_HANDLE target_process, PAL_HANDLE cargo);
int _DkReceiveHandle(PAL_HANDLE source_process, PAL_HANDLE* out_cargo);

/* DkProcess and DkThread calls */
int _DkThreadCreate(PAL_HANDLE* handle, int (*callback)(void*), void* param);
noreturn void _DkThreadExit(int* clear_child_tid);
void _DkThreadYieldExecution(void);
int _DkThreadResume(PAL_HANDLE thread_handle);
int _DkProcessCreate(PAL_HANDLE* handle, const char** args);
noreturn void _DkProcessExit(int exit_code);
int _DkThreadSetCpuAffinity(PAL_HANDLE thread, PAL_NUM cpumask_size, unsigned long* cpu_mask);
int _DkThreadGetCpuAffinity(PAL_HANDLE thread, PAL_NUM cpumask_size, unsigned long* cpu_mask);

/* DkEvent calls */
int _DkEventCreate(PAL_HANDLE* handle_ptr, bool init_signaled, bool auto_clear);
void _DkEventSet(PAL_HANDLE handle);
void _DkEventClear(PAL_HANDLE handle);
int _DkEventWait(PAL_HANDLE handle, uint64_t* timeout_us);

/* DkVirtualMemory calls */
int _DkVirtualMemoryAlloc(void** addr_ptr, uint64_t size, pal_alloc_flags_t alloc_type,
                          pal_prot_flags_t prot);
int _DkVirtualMemoryFree(void* addr, uint64_t size);
int _DkVirtualMemoryProtect(void* addr, uint64_t size, pal_prot_flags_t prot);

/* DkObject calls */
int _DkObjectClose(PAL_HANDLE object_handle);
int _DkStreamsWaitEvents(size_t count, PAL_HANDLE* handle_array, pal_wait_flags_t* events,
                         pal_wait_flags_t* ret_events, uint64_t* timeout_us);

/* DkException calls & structures */
pal_event_handler_t _DkGetExceptionHandler(enum pal_event event);

/* other DK calls */
int _DkSystemTimeQuery(uint64_t* out_usec);

/*
 * Cryptographically secure random.
 * 0 on success, negative on failure.
 */
int _DkRandomBitsRead(void* buffer, size_t size);

double _DkGetBogomips(void);
int _DkSegmentBaseGet(enum pal_segment_reg reg, uintptr_t* addr);
int _DkSegmentBaseSet(enum pal_segment_reg reg, uintptr_t addr);
int _DkCpuIdRetrieve(uint32_t leaf, uint32_t subleaf, uint32_t values[4]);
int _DkDeviceIoControl(PAL_HANDLE handle, unsigned int cmd, uint64_t arg);
int _DkAttestationReport(const void* user_report_data, PAL_NUM* user_report_data_size,
                         void* target_info, PAL_NUM* target_info_size, void* report,
                         PAL_NUM* report_size);
int _DkAttestationQuote(const void* user_report_data, PAL_NUM user_report_data_size, void* quote,
                        PAL_NUM* quote_size);
int _DkGetSpecialKey(const char* name, void* key, size_t* key_size);

#define INIT_FAIL(exitcode, reason)                                                              \
    do {                                                                                         \
        log_error("PAL failed at " __FILE__ ":%s:%u (exitcode = %u, reason=%s)", __FUNCTION__,   \
                  (unsigned int)__LINE__, (unsigned int)(exitcode), reason);                     \
        _DkProcessExit(exitcode);                                                                \
    } while (0)

#define INIT_FAIL_MANIFEST(exitcode, reason)                           \
    do {                                                               \
        log_error("PAL failed at parsing the manifest: %s", reason);   \
        _DkProcessExit(exitcode);                                      \
    } while (0)

void init_slab_mgr(char* mem_pool, size_t mem_pool_size);
void* malloc(size_t size);
void* malloc_copy(const void* mem, size_t size);
void* calloc(size_t num, size_t size);
void free(void* mem);

#ifdef __GNUC__
#define __attribute_hidden        __attribute__((visibility("hidden")))
#define __attribute_always_inline __attribute__((always_inline))
#define __attribute_unused        __attribute__((unused))
#define __attribute_noinline      __attribute__((noinline))
#else
#error Unsupported compiler
#endif

int _DkInitDebugStream(const char* path);
int _DkDebugLog(const void* buf, size_t size);

// TODO(mkow): We should make it cross-object-inlinable, ideally by enabling LTO, less ideally by
// pasting it here and making `inline`, but our current linker scripts prevent both.
void pal_log(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

#define PAL_LOG_DEFAULT_LEVEL  LOG_LEVEL_ERROR
#define PAL_LOG_DEFAULT_FD     2

const char* pal_event_name(enum pal_event event);

#define uthash_fatal(msg)                      \
    do {                                       \
        log_error("uthash error: %s", msg);    \
        _DkProcessExit(PAL_ERROR_NOMEM);       \
    } while (0)
#include "uthash.h"

/* Size of PAL memory available before parsing the manifest; `loader.pal_internal_mem_size` does not
 * include this memory */
#define PAL_INITIAL_MEM_SIZE (64 * 1024 * 1024)

#endif
