/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * A library for dumping synthetic perf.data events, for consumption by 'perf report' tool.
 *
 * For more information on the format, see Linux sources:
 *
 * - tools/perf/Documentation/perf.data-file-format.txt
 * - include/uapi/linux/perf_event.h
 * - tools/perf/util/header.h
 *
 * Unfortunately, the format contains a header with data offsets and sizes, as well as additional
 * headers at the end of files. To simplify processing, we write all of these on closing (pd_close).
 *
 * (There is also a simpler "pipe-mode data" format, which does not require seeking, but it's less
 * convenient to use because perf userspace tools recognize it only when reading from stdin. This
 * has been fixed in Linux 5.8: https://lkml.org/lkml/2020/5/7/294)
 *
 * To view the report, use 'perf report -i <filename>'.
 *
 * For debugging the output, you can use:
 *
 * - perf script -D -i <filename>: show a partial parse of the file
 * - perf script -v -i <filename>: show more errors if the file doesn't parse
 */

#include <asm/errno.h>
#include <asm/perf_regs.h>
#include <assert.h>
#include <linux/perf_event.h>
#include <linux/fs.h>

#include "perm.h"
#include "sgx_internal.h"
#include "sgx_log.h"

#ifndef __x86_64__
#error "Unsupported architecture"
#endif

#define PROLOGUE_SIZE (sizeof(struct perf_file_header) + sizeof(struct perf_file_attr))

/* Buffer size for pending perf data. Choose a big enough size (32 MB) so that we don't need to
 * flush to a file too often. */
#define BUF_SIZE (32 * 1024 * 1024)

/* Registers to sample - see arch/x86/include/uapi/asm/perf_regs.h */
#define SAMPLE_REGS_CNT 18
#define SAMPLE_REGS ((1 << PERF_REG_X86_AX)    | \
                     (1 << PERF_REG_X86_BX)    | \
                     (1 << PERF_REG_X86_CX)    | \
                     (1 << PERF_REG_X86_DX)    | \
                     (1 << PERF_REG_X86_SI)    | \
                     (1 << PERF_REG_X86_DI)    | \
                     (1 << PERF_REG_X86_BP)    | \
                     (1 << PERF_REG_X86_SP)    | \
                     (1 << PERF_REG_X86_IP)    | \
                     (1 << PERF_REG_X86_FLAGS) | \
                     (1 << PERF_REG_X86_R8)    | \
                     (1 << PERF_REG_X86_R9)    | \
                     (1 << PERF_REG_X86_R10)   | \
                     (1 << PERF_REG_X86_R11)   | \
                     (1 << PERF_REG_X86_R12)   | \
                     (1 << PERF_REG_X86_R13)   | \
                     (1 << PERF_REG_X86_R14)   | \
                     (1 << PERF_REG_X86_R15))

/* Internal perf.data file definitions - see linux/tools/perf/util/header.h */

#define PERF_MAGIC 0x32454c4946524550ULL  // "PERFILE2"
#define HEADER_ARCH 6

struct perf_file_section {
    uint64_t offset;
    uint64_t size;
};

struct perf_file_header {
    uint64_t magic;
    uint64_t size;
    uint64_t attr_size;
    struct perf_file_section attrs;
    struct perf_file_section data;
    struct perf_file_section event_types;
    uint64_t flags[4];
};

struct perf_file_attr {
    struct perf_event_attr attr;
    struct perf_file_section ids;
};

struct perf_data {
    int fd;
    // Data to be flushed to a file
    size_t buf_count;
    uint8_t buf[BUF_SIZE];

    bool with_stack;
};

static ssize_t write_all(int fd, const void* buf, size_t count) {
    while (count > 0) {
        ssize_t ret = DO_SYSCALL(write, fd, buf, count);
        if (ret == -EINTR)
            continue;
        if (ret < 0)
            return ret;
        count -= ret;
        buf += ret;
    }
    return 0;
}

static int pd_flush(struct perf_data* pd) {
    if (pd->buf_count == 0)
        return 0;

    ssize_t ret = write_all(pd->fd, pd->buf, pd->buf_count);
    if (ret < 0)
        return ret;
    pd->buf_count = 0;
    return 0;
}

// Add data to buffer; flush first if necessary
static int pd_write(struct perf_data* pd, const void* data, size_t size) {
    if (pd->buf_count + size > sizeof(pd->buf)) {
        int ret = pd_flush(pd);
        if (ret < 0)
            return ret;
    }

    assert(pd->buf_count + size <= sizeof(pd->buf));
    memcpy(pd->buf + pd->buf_count, data, size);
    pd->buf_count += size;
    return 0;
}

struct perf_data* pd_open(const char* file_name, bool with_stack) {
    int ret;

    int fd = DO_SYSCALL(open, file_name, O_WRONLY | O_TRUNC | O_CREAT, PERM_rw_r__r__);
    if (fd < 0) {
        log_error("pd_open: cannot open %s for writing: %d", file_name, fd);
        return NULL;
    }

    /*
     * Start writing to file from PROLOGUE_SIZE position. The beginning will be overwritten in
     * write_prologue_epilogue().
     */

    ret = DO_SYSCALL(ftruncate, fd, PROLOGUE_SIZE);
    if (ret < 0)
        goto fail;

    ret = DO_SYSCALL(lseek, fd, PROLOGUE_SIZE, SEEK_SET);
    if (ret < 0)
        goto fail;

    struct perf_data* pd = malloc(sizeof(*pd));
    if (!pd) {
        log_error("pd_open: out of memory");
        goto fail;
    }

    pd->fd = fd;
    pd->buf_count = 0;
    pd->with_stack = with_stack;
    return pd;

fail:
    ret = DO_SYSCALL(close, fd);
    if (ret < 0)
        log_error("pd_open: close failed: %d", ret);
    return NULL;
};

static int write_prologue_epilogue(struct perf_data* pd) {
    int ret;

    assert(pd->buf_count == 0); // all data flushed
    ssize_t data_end = DO_SYSCALL(lseek, pd->fd, 0, SEEK_CUR);
    if (data_end < 0)
        return data_end;

    /*
     * Prologue, at beginning of file:
     * - struct perf_file_header: description of other file sections
     * - struct perf_file_attr: event configuration
     */

    ret = DO_SYSCALL(lseek, pd->fd, 0, SEEK_SET);
    if (ret < 0)
        return ret;

    // Determines the set of data in PERF_RECORD_SAMPLE
    uint64_t sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_PERIOD;
    if (pd->with_stack)
        sample_type |= PERF_SAMPLE_CALLCHAIN | PERF_SAMPLE_REGS_USER | PERF_SAMPLE_STACK_USER;

    struct perf_file_attr attr = {
        .attr = {
            .type = PERF_TYPE_SOFTWARE,
            .size = sizeof(attr.attr),
            .sample_type = sample_type,
            .sample_regs_user = SAMPLE_REGS,
        },
        .ids = {0},
    };
    struct perf_file_header header = {
        .magic = PERF_MAGIC,
        .size = sizeof(header),
        .attr_size = sizeof(attr.attr),
        .attrs = {
            .offset = sizeof(header),
            .size = sizeof(attr),
        },
        .data = {
            .offset = PROLOGUE_SIZE,
            .size = data_end - PROLOGUE_SIZE,
        },
        .event_types = {0},
        .flags = {0},
    };
    // Signifies that a HEADER_ARCH section will be included after data section
    header.flags[0] |= 1 << HEADER_ARCH;

    ret = write_all(pd->fd, &header, sizeof(header));
    if (ret < 0)
        return ret;
    ret = write_all(pd->fd, &attr, sizeof(attr));
    if (ret < 0)
        return ret;

    /*
     * Epilogue, after the data section. Contains header sections, as specified by header.flags.
     * We include a HEADER_ARCH section. While it's documented as optional, 'perf script' crashes
     * without it.
     */

    ret = DO_SYSCALL(lseek, pd->fd, data_end, SEEK_SET);
    if (ret < 0)
        return ret;

    const char* arch = "x86_64";
    uint32_t arch_size = strlen(arch) + 1;
    struct perf_file_section arch_section = {
        .offset = data_end + sizeof(arch_section),
        .size = sizeof(arch_size) + arch_size,
    };
    ret = write_all(pd->fd, &arch_section, sizeof(arch_section));
    if (ret < 0)
        return ret;
    ret = write_all(pd->fd, &arch_size, sizeof(arch_size));
    if (ret < 0)
        return ret;
    ret = write_all(pd->fd, arch, arch_size);
    if (ret < 0)
        return ret;

    return 0;
}

ssize_t pd_close(struct perf_data* pd) {
    ssize_t ret = 0;
    int close_ret;

    ret = pd_flush(pd);
    if (ret < 0)
        goto out;

    ret = write_prologue_epilogue(pd);
    if (ret < 0)
        goto out;

    ret = DO_SYSCALL(lseek, pd->fd, 0, SEEK_CUR);
    if (ret < 0)
        goto out;

out:
    close_ret = DO_SYSCALL(close, pd->fd);
    if (close_ret < 0)
        log_error("pd_close: close failed: %d", close_ret);

    free(pd);
    return ret;
}

int pd_event_command(struct perf_data* pd, const char* command, uint32_t pid, uint32_t tid) {
    size_t command_size = strlen(command) + 1;
    struct {
        struct perf_event_header header;

        uint32_t pid, tid;
    } event = {
        .header = {
            .type = PERF_RECORD_COMM,
            .misc = 0,
            .size = sizeof(event) + command_size,
        },

        .pid = pid,
        .tid = tid,
    };
    int ret;
    ret = pd_write(pd, &event, sizeof(event));
    if (ret < 0)
        return ret;
    ret = pd_write(pd, command, command_size);
    if (ret < 0)
        return ret;
    return 0;
}

int pd_event_mmap(struct perf_data* pd, const char* filename, uint32_t pid, uint64_t addr,
                  uint64_t len, uint64_t pgoff) {
    size_t filename_size = strlen(filename) + 1;
    struct {
        struct perf_event_header header;

        uint32_t pid, tid;
        uint64_t addr, len, pgoff;
    } event = {
        .header = {
            .type = PERF_RECORD_MMAP,
            .misc = 0,
            .size = sizeof(event) + filename_size,
        },

        .pid = pid,
        .tid = pid,
        .addr = addr,
        .len = len,
        .pgoff = pgoff,
    };
    int ret;
    ret = pd_write(pd, &event, sizeof(event));
    if (ret < 0)
        return ret;
    ret = pd_write(pd, filename, filename_size);
    if (ret < 0)
        return ret;
    return 0;
}

static int pd_event_sample(struct perf_data* pd, uint64_t ip, uint32_t pid, uint32_t tid,
                         uint64_t period, size_t extra_size) {
    struct {
        struct perf_event_header header;

        uint64_t ip;
        uint32_t pid, tid;
        uint64_t period;
    } event = {
        .header = {
            .type = PERF_RECORD_SAMPLE,
            .misc = PERF_RECORD_MISC_USER, // user/kernel/hypervisor
            .size = sizeof(event) + extra_size,
        },

        .ip = ip,
        .pid = pid,
        .tid = tid,
        .period = period,
    };

    return pd_write(pd, &event, sizeof(event));
}

int pd_event_sample_simple(struct perf_data* pd, uint64_t ip, uint32_t pid, uint32_t tid,
                           uint64_t period) {
    assert(!pd->with_stack);
    return pd_event_sample(pd, ip, pid, tid, period, /*extra_size=*/0);
}

int pd_event_sample_stack(struct perf_data* pd, uint64_t ip, uint32_t pid, uint32_t tid,
                          uint64_t period, sgx_pal_gpr_t* gpr, void* stack, size_t stack_size) {
    assert(pd->with_stack);
    struct {
        // Empty callchain section - needed so that perf will attempt to recover call chain
        struct {
            uint64_t nr;
        } callchain;

        struct {
            uint64_t abi;
            uint64_t regs[SAMPLE_REGS_CNT];
        } regs;
    } extra = {
        .callchain = {
            .nr = 0,
        },

        .regs = {
            .abi = PERF_SAMPLE_REGS_ABI_64,
            .regs = {
                gpr->rax,
                gpr->rbx,
                gpr->rcx,
                gpr->rdx,
                gpr->rsi,
                gpr->rdi,
                gpr->rbp,
                gpr->rsp,
                gpr->rip,
                gpr->rflags,
                gpr->r8,
                gpr->r9,
                gpr->r10,
                gpr->r11,
                gpr->r12,
                gpr->r13,
                gpr->r14,
                gpr->r15,
            },
        },
    };
    size_t extra_size = sizeof(extra) + stack_size + 2 * sizeof(uint64_t);
    int ret;

    // Common section
    ret = pd_event_sample(pd, ip, pid, tid, period, extra_size);
    if (ret < 0)
        return ret;

    // Callchain and regs sections
    ret = pd_write(pd, &extra, sizeof(extra));
    if (ret < 0)
        return ret;

    // Stack section (variable length)
    uint64_t size_field = stack_size;
    ret = pd_write(pd, &size_field, sizeof(size_field));  // uint64_t size
    if (ret < 0)
        return ret;
    ret = pd_write(pd, stack, stack_size);
    if (ret < 0)
        return ret;
    ret = pd_write(pd, &size_field, sizeof(size_field));  // uint64_t dyn_size = size
    if (ret < 0)
        return ret;

    return 0;
}
