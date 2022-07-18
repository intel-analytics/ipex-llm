/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */
/* Copyright (C) 2020 Intel Labs */

/*
 * Operations to handle devices (with special case of "dev:tty" which is stdin/stdout).
 *
 * TODO: Some devices allow lseek() but typically with device-specific semantics. Gramine currently
 *       emulates lseek() completely in LibOS layer, thus seeking at PAL layer cannot be correctly
 *       implemented (without device-specific changes to LibOS layer).
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_flags_conv.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "pal_linux_error.h"
#include "perm.h"

static int dev_open(PAL_HANDLE* handle, const char* type, const char* uri, enum pal_access access,
                    pal_share_flags_t share, enum pal_create_mode create,
                    pal_stream_options_t options) {
    int ret;
    assert(create != PAL_CREATE_IGNORED);

    if (strcmp(type, URI_TYPE_DEV))
        return -PAL_ERROR_INVAL;

    assert(WITHIN_MASK(share,   PAL_SHARE_MASK));
    assert(WITHIN_MASK(options, PAL_OPTION_MASK));

    PAL_HANDLE hdl = calloc(1, HANDLE_SIZE(dev));
    if (!hdl)
        return -PAL_ERROR_NOMEM;

    init_handle_hdr(hdl, PAL_TYPE_DEV);

    if (!strcmp(uri, "tty")) {
        /* special case of "dev:tty" device which is the standard input + standard output */
        hdl->dev.nonblocking = false;

        if (access == PAL_ACCESS_RDONLY) {
            hdl->flags |= PAL_HANDLE_FD_READABLE;
            hdl->dev.fd = 0; /* host stdin */
        } else if (access == PAL_ACCESS_WRONLY) {
            hdl->flags |= PAL_HANDLE_FD_WRITABLE;
            hdl->dev.fd = 1; /* host stdout */
        } else {
            assert(access == PAL_ACCESS_RDWR);
            ret = -PAL_ERROR_INVAL;
            goto fail;
        }
    } else {
        /* other devices must be opened through the host */
        hdl->dev.nonblocking = !!(options & PAL_OPTION_NONBLOCK);

        ret = ocall_open(uri, PAL_ACCESS_TO_LINUX_OPEN(access)  |
                              PAL_CREATE_TO_LINUX_OPEN(create)  |
                              PAL_OPTION_TO_LINUX_OPEN(options),
                         share);
        if (ret < 0) {
            ret = unix_to_pal_error(ret);
            goto fail;
        }
        hdl->dev.fd = ret;

        if (access == PAL_ACCESS_RDONLY) {
            hdl->flags |= PAL_HANDLE_FD_READABLE;
        } else if (access == PAL_ACCESS_WRONLY) {
            hdl->flags |= PAL_HANDLE_FD_WRITABLE;
        } else {
            assert(access == PAL_ACCESS_RDWR);
            hdl->flags |= PAL_HANDLE_FD_READABLE | PAL_HANDLE_FD_WRITABLE;
        }
    }

    *handle = hdl;
    return 0;
fail:
    free(hdl);
    return ret;
}

static int64_t dev_read(PAL_HANDLE handle, uint64_t offset, uint64_t size, void* buffer) {
    if (offset || HANDLE_HDR(handle)->type != PAL_TYPE_DEV)
        return -PAL_ERROR_INVAL;

    if (!(handle->flags & PAL_HANDLE_FD_READABLE))
        return -PAL_ERROR_DENIED;

    if (handle->dev.fd == PAL_IDX_POISON)
        return -PAL_ERROR_DENIED;

    ssize_t bytes = ocall_read(handle->dev.fd, buffer, size);
    return bytes < 0 ? unix_to_pal_error(bytes) : bytes;
}

static int64_t dev_write(PAL_HANDLE handle, uint64_t offset, uint64_t size, const void* buffer) {
    if (offset || HANDLE_HDR(handle)->type != PAL_TYPE_DEV)
        return -PAL_ERROR_INVAL;

    if (!(handle->flags & PAL_HANDLE_FD_WRITABLE))
        return -PAL_ERROR_DENIED;

    if (handle->dev.fd == PAL_IDX_POISON)
        return -PAL_ERROR_DENIED;

    ssize_t bytes = ocall_write(handle->dev.fd, buffer, size);
    return bytes < 0 ? unix_to_pal_error(bytes) : bytes;
}

static int dev_close(PAL_HANDLE handle) {
    if (HANDLE_HDR(handle)->type != PAL_TYPE_DEV)
        return -PAL_ERROR_INVAL;

    /* currently we just assign `0`/`1` FDs without duplicating, so close is a no-op for them */
    int ret = 0;
    if (handle->dev.fd != PAL_IDX_POISON && handle->dev.fd != 0 && handle->dev.fd != 1) {
        ret = ocall_close(handle->dev.fd);
    }
    handle->dev.fd = PAL_IDX_POISON;
    return ret < 0 ? unix_to_pal_error(ret) : 0;
}

static int dev_flush(PAL_HANDLE handle) {
    if (HANDLE_HDR(handle)->type != PAL_TYPE_DEV)
        return -PAL_ERROR_INVAL;

    if (handle->dev.fd != PAL_IDX_POISON) {
        int ret = ocall_fsync(handle->dev.fd);
        if (ret < 0)
            return unix_to_pal_error(ret);
    }
    return 0;
}

static int dev_attrquery(const char* type, const char* uri, PAL_STREAM_ATTR* attr) {
    __UNUSED(uri);

    if (strcmp(type, URI_TYPE_DEV))
        return -PAL_ERROR_INVAL;

    if (!strcmp(uri, "tty")) {
        /* special case of "dev:tty" device which is the standard input + standard output */
        attr->share_flags  = PERM_rw_rw_rw_;
        attr->pending_size = 0;
    } else {
        /* other devices must query the host */
        int fd = ocall_open(uri, 0, 0);
        if (fd < 0)
            return unix_to_pal_error(fd);

        struct stat stat_buf;
        int ret = ocall_fstat(fd, &stat_buf);
        if (ret < 0) {
            ocall_close(fd);
            return unix_to_pal_error(ret);
        }

        attr->share_flags  = stat_buf.st_mode & PAL_SHARE_MASK;
        attr->pending_size = stat_buf.st_size;

        ocall_close(fd);
    }

    attr->handle_type  = PAL_TYPE_DEV;
    attr->nonblocking  = false;
    return 0;
}

static int dev_attrquerybyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    if (HANDLE_HDR(handle)->type != PAL_TYPE_DEV)
        return -PAL_ERROR_INVAL;

    if (handle->dev.fd == 0 || handle->dev.fd == 1) {
        /* special case of "dev:tty" device which is the standard input + standard output */
        attr->share_flags  = 0;
        attr->pending_size = 0;
    } else {
        /* other devices must query the host */
        struct stat stat_buf;
        int ret = ocall_fstat(handle->dev.fd, &stat_buf);
        if (ret < 0)
            return unix_to_pal_error(ret);

        attr->share_flags  = stat_buf.st_mode & PAL_SHARE_MASK;
        attr->pending_size = stat_buf.st_size;
    }

    attr->handle_type  = PAL_TYPE_DEV;
    attr->nonblocking  = handle->dev.nonblocking;
    return 0;
}

/* this dummy function is implemented to support opening TTY devices with O_TRUNC flag */
static int64_t dev_setlength(PAL_HANDLE handle, uint64_t length) {
    if (HANDLE_HDR(handle)->type != PAL_TYPE_DEV)
        return -PAL_ERROR_INVAL;

    if (!(handle->dev.fd == 0 || handle->dev.fd == 1))
        return -PAL_ERROR_NOTSUPPORT;

    if (length != 0)
        return -PAL_ERROR_INVAL;

    return 0;
}

struct handle_ops g_dev_ops = {
    .open           = &dev_open,
    .read           = &dev_read,
    .write          = &dev_write,
    .close          = &dev_close,
    .setlength      = &dev_setlength,
    .flush          = &dev_flush,
    .attrquery      = &dev_attrquery,
    .attrquerybyhdl = &dev_attrquerybyhdl,
};


/*
 * Code below describes the deep-copy syntax in the TOML manifest used for copying complex nested
 * objects out and in the SGX enclave. This syntax is currently used for IOCTL emulation. This
 * syntax is generic enough to describe any memory layout for deep copy.
 *
 * The following example describes the main implementation details:
 *
 *   struct pascal_str { uint8_t len; char str[]; };
 *   struct c_str { char str[]; };
 *   struct root { struct pascal_str* s1; struct c_str* s2; uint64_t s2_len; int8_t x; int8_t y; };
 *
 *   alignas(128) struct root obj;
 *   ioctl(devfd, _IOWR(DEVICE_MAGIC, DEVICE_FUNC, struct root), &obj);
 *
 * The example IOCTL takes as a third argument a pointer to an object of type `struct root` that
 * contains two pointers to other objects (pascal-style string and a C-style string) and embeds two
 * integers `x` and `y`. The two strings reside in separate memory regions in enclave memory. Note
 * that the length of the C-style string is stored in the `s2_len` field of the root object. The
 * `pascal_str` string is an input to the IOCTL, the `c_str` string is both input and output of the
 * IOCTL, and the integers `x` and `y` are both outputs of the IOCTL. Also note that the root
 * object is 128B-aligned (for illustration purposes).
 *
 * The corresponding deep-copy syntax in TOML looks like this:
 *
 *   sgx.ioctl_structs.ROOT_FOR_DEVICE_FUNC = [
 *     { align = 128, ptr = [ {name="pascal-str-len", size=1, type="out"},
 *                            {name="pascal-str", size="pascal-str-len", type="out"} ] },
 *     { ptr = [ {name="c-str", size="c-str-len", type="inout"} ], size = 1 },
 *     { name = "c-str-len", size = 8, unit = 1, adjust = 0, type = "in" },
 *     { onlyif = "c-str-len == pascal-str-len", size = 2, type = "in" }
 *     { onlyif = "c-str-len != pascal-str-len", size = 2, type = "none" }
 *   ]
 *
 * One can observe the following rules in this TOML syntax:
 *  1. Each separate memory region is represented as a TOML array (`[]`).
 *  2. Each sub-region of one memory region is represented as a TOML table (`{}`).
 *  3. Each sub-region may be a pointer (`ptr`) to another memory region. In this case, the value of
 *     `ptr` is a TOML-array representation of that other memory region. The `ptr` sub-region always
 *     has size of 8B (assuming x86-64) and doesn't have an in/out type. The `size` field of the
 *     `ptr` sub-region has a different meaning than for non-pointer sub-regions: it is the number
 *     of adjacent memory regions that this pointer points to (i.e. it describes an array).
 *  4. Sub-regions can be fixed-size (like the last sub-region containing two bytes `x` and `y`) or
 *     can be flexible-size (like the two strings). In the latter case, the `size` field contains a
 *     name of a sub-region where the actual size is stored.
 *  5. Sub-regions that store the size of another sub-region must be 1, 2, 4, or 8 bytes in size.
 *  6. Sub-regions may have a name for ease of identification; this is required for "size"
 *     sub-regions but may be omitted for all other kinds of sub-regions.
 *  7. Sub-regions may have one of the four types: "out" to copy contents of the sub-region outside
 *     the enclave to untrusted memory, "in" to copy from untrusted memory to inside the enclave,
 *     "inout" to copy in both directions, "none" to not copy at all (useful for e.g. padding).
 *     Note that pointer sub-regions do not have a type.
 *  8. The first sub-region (and only the first!) may specify the alignment of the memory region.
 *  9. The total size of a sub-region is calculated as `size * unit + adjust`. By default `unit` is
 *     1 byte and `adjust` is 0. Note that `adjust` may be a negative number.
 * 10. Sub-regions may be conditioned using `onlyif = "simple boolean expression"`. In the example
 *     above, `x` and `y` will be copied back into the enclave only if the two strings have the
 *     same length; otherwise `x` and `y` will not be modified. The only currently supported format
 *     of expressions are (`token1` and `token2` may be constant integers or sub region names):
 *     - "token1 == token2"
 *     - "token1 != token2"
 *     - "token1 &= token2" (all bits set, same as `token1 & token2 == token2` in C)
 *     - "token1 |= token2" (at least one bit set, same as `token1 & token2 != 0` in C)
 *
 * The diagram below shows how this complex object is copied from enclave memory (left side) to
 * untrusted memory (right side). MR stands for "memory region", SR stands for "sub-region". Note
 * how enclave pointers are copied and rewired to point to untrusted memory regions.
 *
 *       struct root (MR1)            |       deep-copied struct (aligned at 128B)
 *      +------------------+          |     +------------------------+
 *  +----+ pascal_str* s1  |     SR1  |  +----+ pascal_str* s1  (MR1)|
 *  |   |                  |          |  |  |                        |
 *  |   |  c_str* s2 +-------+   SR2  |  |  |   c_str* s2 +-------------+
 *  |   |                  | |        |  |  |                        |  |
 *  |   |  uint64_t s2_len | |   SR3  |  |  |   uint64_t s2_len      |  |
 *  |   |                  | |        |  |  |                        |  |
 *  |   |  int8_t x, y     | |   SR4  |  |  |   int8_t x=0, y=0      |  |
 *  |   +------------------+ |        |  |  +------------------------+  |
 *  |                        |        |  +->|   uint8_t len     (MR2)|  |
 *  v (MR2)                  |        |     |                        |  |
 * +-------------+           |        |     |   char str[]           |  |
 * | uint8_t len |           |   SR5  |     +------------------------+  |
 * |             |           |        |     |   char str[]      (MR3)|<-+
 * | char str[]  |           |   SR6  |     +------------------------+
 * +-------------+           |        |
 *                  (MR3)    v        |
 *                +----------+-+      |
 *                | char str[] | SR7  |
 *                +------------+      |
 *
 */

/* for simplicity, we allocate limited number of mem_regions on stack */
#define MAX_MEM_REGIONS 1024

/* for simplicity, we allocate limited number of sub_regions on heap */
#define MAX_SUB_REGIONS (10 * 1024)

#define IOCTL_STRUCT_CACHE_SIZE 32
#define SUB_REGION_INFO_CACHE_SIZE 64

/* simple hash function djb2 by Dan Bernstein */
static uint64_t hash(char *str) {
    uint64_t hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
    return hash;
}

/* direction of copy: none (used for padding), out of enclave, inside enclave, both or a special
 * "pointer" sub-region; default is COPY_NONE_ENCLAVE */
enum mem_copy_type {COPY_NONE_ENCLAVE = 0, COPY_OUT_ENCLAVE, COPY_IN_ENCLAVE, COPY_INOUT_ENCLAVE,
                    COPY_PTR_ENCLAVE};

struct mem_region {
    toml_array_t* toml_array; /* describes contigious sub_regions in this mem_region */
    void* encl_addr;          /* base address of this memory region in enclave memory */
    bool adjacent;            /* memory region adjacent to previous one? (used for arrays) */
};

struct sub_region {
    enum mem_copy_type type; /* direction of copy during OCALL (or pointer to another region) */
    char* name;              /* may be NULL for unnamed regions */
    uint64_t name_hash;      /* hash of "name" for fast string comparison */
    bool name_need_free;     /* false if "name" ownership was transfered to the cache */
    ssize_t align;           /* alignment of this sub-region */
    ssize_t size;            /* may be dynamically determined from another sub-region */
    char* size_name;         /* needed if "size" sub region is defined after this sub region */
    uint64_t size_name_hash; /* needed if "size" sub region is defined after this sub region */
    bool size_name_need_free;/* false if "size_name" ownership was transfered to the cache */
    ssize_t unit;            /* total size in bytes is calculated as `size * unit` */
    ssize_t adjust;          /* may be negative; total size in bytes is `size * unit + adjust` */
    void* encl_addr;         /* base address of this sub region in enclave memory */
    void* untrusted_addr;    /* base address of the corresponding sub region in untrusted memory */
    toml_array_t* mem_ptr;   /* for pointers/arrays, specifies pointed-to mem region */
};

struct cached_ioctl_struct {
    unsigned int cmd;
    toml_array_t* toml_array;
};

struct cached_sub_region_info {
    /* lookup in the cache happens on this main pointer */
    toml_table_t* toml_table;

    /* pointers to raw fields of the sub region */
    toml_raw_t onlyif_raw;
    toml_raw_t name_raw;
    toml_raw_t type_raw;
    toml_raw_t align_raw;
    toml_raw_t size_raw;
    toml_raw_t unit_raw;
    toml_raw_t adjust_raw;

    /* pointer to a nested memory region */
    toml_array_t* ptr_arr;

    /* sub region's cached values */
    enum mem_copy_type type;
    char* name;
    uint64_t name_hash;
    ssize_t size; /* -1 if dynamically determined based on name, then use size_name */
    char* size_name;
    uint64_t size_name_hash;
    ssize_t align;
    ssize_t unit;
    ssize_t adjust;
};

static bool strings_equal(const char* s1, const char* s2, uint64_t s1_hash, uint64_t s2_hash) {
    if (!s1 || !s2 || s1_hash != s2_hash)
        return false;
    assert(s1_hash == s2_hash);
    return !strcmp(s1, s2);
}

/* finds a sub region with name `sub_region_name` among `sub_regions` and return its index */
static int get_sub_region_idx(struct sub_region* sub_regions, int sub_regions_cnt,
                              const char* sub_region_name, uint64_t sub_region_name_hash,
                              int* idx) {
    /* it is important to iterate in reverse order because there may be an array of mem regions
     * with same-named sub regions, and we want to find the latest sub region */
    for (int i = sub_regions_cnt - 1; i >= 0; i--) {
        if (strings_equal(sub_regions[i].name, sub_region_name,
                          sub_regions[i].name_hash, sub_region_name_hash)) {
            /* found corresponding sub region */
            if (sub_regions[i].type != COPY_PTR_ENCLAVE || !sub_regions[i].mem_ptr) {
                /* sub region is not a valid pointer to a memory region */
                return -EFAULT;
            }
            *idx = i;
            return 0;
        }
    }
    return -ENOENT;
}

/* finds a sub region with name `sub_region_name` among `sub_regions` and reads the value in it */
static int get_sub_region_value(struct sub_region* sub_regions, int sub_regions_cnt,
                                const char* sub_region_name, uint64_t sub_region_name_hash,
                                ssize_t* value) {
    /* it is important to iterate in reverse order because there may be an array of mem regions
     * with same-named sub regions, and we want to find the "latest value" sub region, i.e. the one
     * belonging to the same mem region */
    for (int i = sub_regions_cnt - 1; i >= 0; i--) {
        if (strings_equal(sub_regions[i].name, sub_region_name,
                          sub_regions[i].name_hash, sub_region_name_hash)) {
            /* found corresponding sub region, read its value */
            if (!sub_regions[i].encl_addr || sub_regions[i].encl_addr == (void*)-1) {
                /* enclave address is invalid, user provided bad struct */
                return -EFAULT;
            }

            if (sub_regions[i].size == sizeof(uint8_t)) {
                *value = (ssize_t)(*((uint8_t*)sub_regions[i].encl_addr));
            } else if (sub_regions[i].size == sizeof(uint16_t)) {
                *value = (ssize_t)(*((uint16_t*)sub_regions[i].encl_addr));
            } else if (sub_regions[i].size == sizeof(uint32_t)) {
                *value = (ssize_t)(*((uint32_t*)sub_regions[i].encl_addr));
            } else if (sub_regions[i].size == sizeof(uint64_t)) {
                *value = (ssize_t)(*((uint64_t*)sub_regions[i].encl_addr));
            } else {
                log_error("Invalid deep-copy syntax (deep-copy sub-entry \'%s\' must be of "
                               "legitimate size: 1, 2, 4 or 8 bytes)\n",
                               sub_regions[i].name);
                return -EINVAL;
            }

            return 0;
        }
    }

    return -ENOENT;
}

/* Parses a simple boolean expression in `expr`, finds corresponding values from sub regions and
 * calculates the resulting boolean value. NOTE: currently supports only "x == y", "x != y",
 * "x &= y" (all bits of y set in x) and "x |= y" (at least one bit of y set in x) where both "x"
 * and "y" can be sub region names or constant integers. */
static int calculate_boolean_expr(struct sub_region* sub_regions, int sub_regions_cnt, char* expr,
                                  bool* value) {
    int ret;
    char* cur = expr;

    while (*cur == ' ' || *cur == '\t') { cur++; }

    /* read first token */
    char* token1 = cur;
    while (isalnum(*cur) || *cur == '_' || *cur == '-') { cur++; }
    size_t token1_len = cur - token1;

    while (*cur == ' ' || *cur == '\t') { cur++; }

    /* read comparator */
    char* compar = cur;
    while (*cur == '=' || *cur == '!' || *cur == '&' || *cur == '|') { cur++; }
    size_t compar_len = cur - compar;

    while (*cur == ' ' || *cur == '\t') { cur++; }

    /* read second token */
    char* token2 = cur;
    while (isalnum(*cur) || *cur == '_' || *cur == '-') { cur++; }
    size_t token2_len = cur - token2;

    while (*cur == ' ' || *cur == '\t') { cur++; }

    /* make sure the whole string is in allowed format "token1 {==, !=, &=, |=} token2" */
    if (compar_len != 2 || (memcmp(compar, "==", 2) && memcmp(compar, "!=", 2) &&
                            memcmp(compar, "&=", 2) && memcmp(compar, "|=", 2))) {
        log_error("Invalid deep-copy syntax (the only supported comparators are '==', '!=', "
                       "'&=' and '!=' but they are not found in expression \'%s\')\n", expr);
        return -EINVAL;
    }

    if (*cur != '\0' || !token1_len || !token2_len) {
        log_error("Invalid deep-copy syntax (cannot parse expression \'%s\')\n", expr);
        return -EINVAL;
    }

    /* get actual values for two tokens */
    char* endptr = NULL;
    token1[token1_len] = '\0';
    ssize_t val1 = strtol(token1, &endptr, /*base=*/0);
    if (endptr != token1 + token1_len) {
        /* could not read the constant integer, the token must be a string-name of a sub region */
        ret = get_sub_region_value(sub_regions, sub_regions_cnt, token1, hash(token1), &val1);
        if (ret < 0) {
            log_error("Invalid deep-copy syntax (cannot find first sub region in expression "
                           "\'%s\')\n", expr);
            return ret;
        }
    }

    token2[token2_len] = '\0';
    ssize_t val2 = strtol(token2, &endptr, /*base=*/0);
    if (endptr != token2 + token2_len) {
        /* could not read the constant integer, the token must be a string-name of a sub region */
        ret = get_sub_region_value(sub_regions, sub_regions_cnt, token2, hash(token2), &val2);
        if (ret < 0) {
            log_error("Invalid deep-copy syntax (cannot find second sub region in expression "
                           "\'%s\')\n", expr);
            return ret;
        }
    }

    if (!(memcmp(compar, "==", 2))) {
        *value = val1 == val2;
    } else if (!(memcmp(compar, "!=", 2))) {
        *value = val1 != val2;
    } else if (!(memcmp(compar, "&=", 2))) {
        *value = (val1 & val2) == val2;  /* all bits set */
    } else if (!(memcmp(compar, "|=", 2))) {
        *value = (val1 & val2) != 0;     /* at least one bit set */
    } else {
        assert(0); /* impossible because we verify compar above */
    }
    return 0;
}

/* caller sets `_sub_regions_cnt` to maximum number of sub_regions; this variable is updated to
 * return the number of actually used sub_regions */
static int collect_sub_regions(toml_array_t* root_toml_array, void* root_encl_addr,
                               struct sub_region* sub_regions, int* _sub_regions_cnt) {
    int ret;

    assert(root_toml_array && toml_array_nelem(root_toml_array) > 0);
    assert(sub_regions && _sub_regions_cnt);

    int max_sub_regions = *_sub_regions_cnt;
    int sub_regions_cnt = 0;

    struct mem_region mem_regions[MAX_MEM_REGIONS];
    mem_regions[0].toml_array = root_toml_array;
    mem_regions[0].encl_addr  = root_encl_addr;
    mem_regions[0].adjacent   = false;
    int mem_regions_cnt = 1;

    assert(get_tcb_trts()->ioctl_scratch_space);
    char* cptr = (char*)get_tcb_trts()->ioctl_scratch_space +
                      MAX_SUB_REGIONS * sizeof(struct sub_region) +
                      IOCTL_STRUCT_CACHE_SIZE * sizeof(struct cached_ioctl_struct);
    struct cached_sub_region_info* sub_region_info_cache = (struct cached_sub_region_info*)cptr;

    /* collecting memory regions and their sub-regions must use breadth-first search to dynamically
     * calculate sizes of sub-regions even if they are specified via another sub-region's "name" */
    char* cur_encl_addr = NULL; /* char* type for pointer arithmetic */
    int mem_region_idx = 0;
    while (mem_region_idx < mem_regions_cnt) {
        struct mem_region* cur_mem_region = &mem_regions[mem_region_idx];
        mem_region_idx++;

        if (!cur_mem_region->adjacent)
            cur_encl_addr = cur_mem_region->encl_addr;

        int cur_mem_region_first_sub_region = sub_regions_cnt;

        for (int i = 0; i < toml_array_nelem(cur_mem_region->toml_array); i++) {
            toml_table_t* sub_region_info = toml_table_at(cur_mem_region->toml_array, i);
            if (!sub_region_info) {
                log_error("Invalid deep-copy syntax (each memory subregion must be a TOML "
                               "table)\n");
                ret = -EINVAL;
                goto out;
            }

            if (sub_regions_cnt == max_sub_regions) {
                log_error("Too many memory sub-regions in a deep-copy syntax (maximum "
                               "possible is %d)\n", max_sub_regions);
                ret = -ENOMEM;
                goto out;
            }

            struct sub_region* cur_sub_region = &sub_regions[sub_regions_cnt];
            sub_regions_cnt++;

            cur_sub_region->untrusted_addr = NULL;
            cur_sub_region->mem_ptr = NULL;

            cur_sub_region->encl_addr = cur_encl_addr;
            if (!cur_encl_addr || cur_encl_addr == (void*)-1) {
                /* enclave address is invalid, user provided bad struct */
                ret = -EFAULT;
                goto out;
            }

            struct cached_sub_region_info* cached = NULL;

            for (size_t k = 0; k < SUB_REGION_INFO_CACHE_SIZE; k++) {
                if (!sub_region_info_cache[k].toml_table) {
                    /* the rest of cache is uninitialized, no sense in checking it */
                    break;
                }
                if (sub_region_info_cache[k].toml_table == sub_region_info) {
                    cached = &sub_region_info_cache[k];
                    break;
                }
            }

            toml_raw_t sub_region_onlyif_raw = cached ? cached->onlyif_raw
                                                      : toml_raw_in(sub_region_info, "onlyif");
            toml_raw_t sub_region_name_raw   = cached ? cached->name_raw
                                                      : toml_raw_in(sub_region_info, "name");
            toml_raw_t sub_region_type_raw   = cached ? cached->type_raw
                                                      : toml_raw_in(sub_region_info, "type");
            toml_raw_t sub_region_align_raw  = cached ? cached->align_raw
                                                      : toml_raw_in(sub_region_info, "align");
            toml_raw_t sub_region_size_raw   = cached ? cached->size_raw
                                                      : toml_raw_in(sub_region_info, "size");
            toml_raw_t sub_region_unit_raw   = cached ? cached->unit_raw
                                                      : toml_raw_in(sub_region_info, "unit");
            toml_raw_t sub_region_adjust_raw = cached ? cached->adjust_raw
                                                      : toml_raw_in(sub_region_info, "adjust");

            toml_array_t* sub_region_ptr_arr = cached ? cached->ptr_arr
                                                      : toml_array_in(sub_region_info, "ptr");
            if (!cached && !sub_region_ptr_arr) {
                /* "ptr" to another sub-region is not cached and also doesn't use TOML's inline
                 * array syntax, maybe it is a reference to already-defined sub-region (e.g.,
                 * `ptr = "my-struct"`) */
                toml_raw_t sub_region_ptr_raw = toml_raw_in(sub_region_info, "ptr");
                if (sub_region_ptr_raw) {
                    char* sub_region_name = NULL;
                    ret = toml_rtos(sub_region_ptr_raw, &sub_region_name);
                    if (ret < 0) {
                        log_error("Invalid deep-copy syntax (\'ptr\' of a deep-copy sub-entry "
                                "must be a TOML array or a string surrounded by double quotes)\n");
                        ret = -EINVAL;
                        goto out;
                    }

                    int idx = -1;
                    ret = get_sub_region_idx(sub_regions, sub_regions_cnt, sub_region_name,
                                             hash(sub_region_name), &idx);
                    if (ret < 0) {
                        log_error("Invalid deep-copy syntax (cannot find sub region \'%s\')\n",
                                sub_region_name);
                        free(sub_region_name);
                        goto out;
                    }

                    assert(idx >= 0 && idx < sub_regions_cnt);
                    assert(sub_regions[idx].type == COPY_PTR_ENCLAVE && sub_regions[idx].mem_ptr);

                    sub_region_ptr_arr = sub_regions[idx].mem_ptr;
                    free(sub_region_name);
                }
            }

            if (sub_region_align_raw && i != 0) {
                log_error("Invalid deep-copy syntax (\'align\' may be specified only for the "
                               "first sub-region of the memory region)\n");
                ret = -EINVAL;
                goto out;
            }

            if (sub_region_type_raw && sub_region_ptr_arr) {
                log_error("Invalid deep-copy syntax (\'ptr\' sub-entries cannot specify "
                               "a \'type\'; pointers are never copied directly but rewired)\n");
                ret = -EINVAL;
                goto out;
            }

            if (sub_region_onlyif_raw) {
                char* onlyif_expr = NULL;
                ret = toml_rtos(sub_region_onlyif_raw, &onlyif_expr);
                if (ret < 0) {
                    log_error("Invalid deep-copy syntax (\'onlyif\' of a deep-copy sub-entry "
                                    "must be a TOML string surrounded by double quotes)\n");
                    ret = -EINVAL;
                    goto out;
                }

                bool val = false;
                /* "sub_regions_cnt - 1" is to exclude myself */
                ret = calculate_boolean_expr(sub_regions, sub_regions_cnt - 1, onlyif_expr, &val);
                if (ret < 0) {
                    free(onlyif_expr);
                    goto out;
                }
                free(onlyif_expr);

                if (!val) {
                    /* onlyif expression is false, we must skip this sub region completely */
                    sub_regions_cnt--;
                    continue;
                }
            }

            cur_sub_region->name = NULL;
            cur_sub_region->name_hash = 0;
            cur_sub_region->name_need_free = false;
            if (cached) {
                cur_sub_region->name = cached->name;
                cur_sub_region->name_hash = cached->name_hash;
            } else if (sub_region_name_raw) {
                ret = toml_rtos(sub_region_name_raw, &cur_sub_region->name);
                if (ret < 0) {
                    log_error("Invalid deep-copy syntax (\'name\' of a deep-copy sub-entry "
                                    "must be a TOML string surrounded by double quotes)\n");
                    ret = -EINVAL;
                    goto out;
                }
                cur_sub_region->name_hash = hash(cur_sub_region->name);
                cur_sub_region->name_need_free = true;
            }

            cur_sub_region->type = COPY_NONE_ENCLAVE;
            if (cached) {
                cur_sub_region->type = cached->type;
            } else if (sub_region_type_raw) {
                char* type_str = NULL;
                ret = toml_rtos(sub_region_type_raw, &type_str);
                if (ret < 0) {
                    log_error("Invalid deep-copy syntax (\'type\' of a deep-copy sub-entry "
                                    "must be a TOML string surrounded by double quotes)\n");
                    ret = -EINVAL;
                    goto out;
                }

                if (!strcmp(type_str, "out")) {
                    cur_sub_region->type = COPY_OUT_ENCLAVE;
                }
                else if (!strcmp(type_str, "in")) {
                    cur_sub_region->type = COPY_IN_ENCLAVE;
                }
                else if (!strcmp(type_str, "inout")) {
                    cur_sub_region->type = COPY_INOUT_ENCLAVE;
                }
                else if (!strcmp(type_str, "none")) {
                    cur_sub_region->type = COPY_NONE_ENCLAVE;
                }
                else {
                    log_error("Invalid deep-copy syntax (\'type\' of a deep-copy sub-entry "
                                    "must be one of \"out\", \"in\", \"inout\" or \"none\")\n");
                    free(type_str);
                    ret = -EINVAL;
                    goto out;
                }

                free(type_str);
            }

            cur_sub_region->align = 0;
            if (cached) {
                cur_sub_region->align = cached->align;
            } else if (sub_region_align_raw) {
                ret = toml_rtoi(sub_region_align_raw, &cur_sub_region->align);
                if (ret < 0 || cur_sub_region->align <= 0) {
                    log_error("Invalid deep-copy syntax (\'align\' of a deep-copy sub-entry "
                                   "must be a positive number)\n");
                    ret = -EINVAL;
                    goto out;
                }
            }

            if (sub_region_ptr_arr) {
                /* only set type for now, we postpone pointer/array handling for later */
                cur_sub_region->type    = COPY_PTR_ENCLAVE;
                cur_sub_region->mem_ptr = sub_region_ptr_arr;
            }

            cur_sub_region->size = -1;
            cur_sub_region->size_name = NULL;
            cur_sub_region->size_name_hash = 0;
            cur_sub_region->size_name_need_free = false;
            if (cached && cached->size_name) {
                /* "size" is dynamically calculated, use cached "size_name" */
                cur_sub_region->size_name = cached->size_name;
                cur_sub_region->size_name_hash = cached->size_name_hash;

                ssize_t val = -1;
                /* "sub_regions_cnt - 1" is to exclude myself; do not fail if couldn't find
                 * (we will try later one more time) */
                ret = get_sub_region_value(sub_regions, sub_regions_cnt - 1,
                                           cur_sub_region->size_name,
                                           cur_sub_region->size_name_hash, &val);
                if (ret < 0 && ret != -ENOENT) {
                    goto out;
                }
                cur_sub_region->size = val;
            } else if (cached && cached->size > 0) {
                /* "size" is static, value simply cached */
                cur_sub_region->size = cached->size;
            } else if (sub_region_size_raw) {
                ret = toml_rtos(sub_region_size_raw, &cur_sub_region->size_name);
                if (ret == 0) {
                    cur_sub_region->size_name_hash = hash(cur_sub_region->size_name);
                    cur_sub_region->size_name_need_free = true;

                    ssize_t val = -1;
                    /* "sub_regions_cnt - 1" is to exclude myself; do not fail if couldn't find
                     * (we will try later one more time) */
                    ret = get_sub_region_value(sub_regions, sub_regions_cnt - 1,
                                               cur_sub_region->size_name,
                                               cur_sub_region->size_name_hash, &val);
                    if (ret < 0 && ret != -ENOENT) {
                        goto out;
                    }
                    cur_sub_region->size = val;
                } else {
                    /* size is specified not as string (another sub-region's name), then must be
                     * specified explicitly as number of bytes */
                    ret = toml_rtoi(sub_region_size_raw, &cur_sub_region->size);
                    if (ret < 0 || cur_sub_region->size <= 0) {
                        log_error("Invalid deep-copy syntax (\'size\' of a deep-copy "
                                       "sub-entry must be a TOML string or a positive number)\n");
                        ret = -EINVAL;
                        goto out;
                    }
                }
            }

            cur_sub_region->unit = 1; /* 1 byte by default */
            if (cached) {
                cur_sub_region->unit = cached->unit;
            } else if (sub_region_unit_raw) {
                ret = toml_rtoi(sub_region_unit_raw, &cur_sub_region->unit);
                if (ret < 0 || cur_sub_region->unit <= 0) {
                    log_error("Invalid deep-copy syntax (\'unit\' of a deep-copy sub-entry "
                                   "must be a positive number)\n");
                    ret = -EINVAL;
                    goto out;
                }
            }

            cur_sub_region->adjust = 0;
            if (cached) {
                cur_sub_region->adjust = cached->adjust;
            } else if (sub_region_adjust_raw) {
                ret = toml_rtoi(sub_region_adjust_raw, &cur_sub_region->adjust);
                if (ret < 0) {
                    log_error("Invalid deep-copy syntax (\'adjust\' of a deep-copy sub-entry "
                                   "is not a valid number)\n");
                    ret = -EINVAL;
                    goto out;
                }
            }

            if (!cached) {
                /* the cache of sub regions uses a naive FIFO eviction algorithm; ideally we should
                 * use LRU but FIFO works fine for current IOCTL workloads */
                struct cached_sub_region_info* new_cached = &sub_region_info_cache[
                    get_tcb_trts()->ioctl_scratch_reg % SUB_REGION_INFO_CACHE_SIZE];

                /* free last cached names in this slot if any */
                free(new_cached->name);
                free(new_cached->size_name);

                new_cached->toml_table     = sub_region_info;
                new_cached->onlyif_raw     = sub_region_onlyif_raw;
                new_cached->name_raw       = sub_region_name_raw;
                new_cached->type_raw       = sub_region_type_raw;
                new_cached->align_raw      = sub_region_align_raw;
                new_cached->size_raw       = sub_region_size_raw;
                new_cached->unit_raw       = sub_region_unit_raw;
                new_cached->adjust_raw     = sub_region_adjust_raw;
                new_cached->ptr_arr        = sub_region_ptr_arr;
                new_cached->name           = cur_sub_region->name;
                new_cached->name_hash      = cur_sub_region->name_hash;
                new_cached->size_name      = cur_sub_region->size_name;
                new_cached->size_name_hash = cur_sub_region->size_name_hash;
                new_cached->type           = cur_sub_region->type;
                new_cached->align          = cur_sub_region->align;
                new_cached->unit           = cur_sub_region->unit;
                new_cached->adjust         = cur_sub_region->adjust;

                new_cached->size = -1;
                if (cur_sub_region->size > 0 && !cur_sub_region->size_name) {
                    /* the size is static (and positive) so we can cache it */
                    new_cached->size = cur_sub_region->size;
                }

                /* pointer ownership of name/size_name is transfered to new_cached */
                cur_sub_region->name_need_free      = false;
                cur_sub_region->size_name_need_free = false;

                get_tcb_trts()->ioctl_scratch_reg++;
            }

            if (cur_sub_region->size >= 0) {
                cur_sub_region->size *= cur_sub_region->unit;
                cur_sub_region->size += cur_sub_region->adjust;
            }

            cur_encl_addr += cur_sub_region->type == COPY_PTR_ENCLAVE ? 8 : cur_sub_region->size;
        }

        /* iterate through collected pointer/array sub regions and add corresponding mem regions */
        for (int i = cur_mem_region_first_sub_region; i < sub_regions_cnt; i++) {
            if (sub_regions[i].type != COPY_PTR_ENCLAVE)
                continue;

            if (sub_regions[i].size >= 0) {
                /* sizes was found in the first swoop, nothing to do here */
            } else if (sub_regions[i].size < 0 && sub_regions[i].size_name) {
                /* pointer/array size was not found in the first swoop, try again */
                ssize_t val = -1;
                ret = get_sub_region_value(sub_regions, sub_regions_cnt, sub_regions[i].size_name,
                                           sub_regions[i].size_name_hash, &val);
                if (ret < 0) {
                    log_error("Invalid deep-copy syntax (cannot find sub region \'%s\')\n",
                            sub_regions[i].size_name);
                    goto out;
                }
                sub_regions[i].size = val;
            } else {
                /* size is not specified at all for this sub region, assume it is 1 item */
                sub_regions[i].size = 1;
            }

            for (ssize_t k = 0; k < sub_regions[i].size; k++) {
                if (mem_regions_cnt == MAX_MEM_REGIONS) {
                    log_error("Too many memory regions in a deep-copy syntax (maximum "
                            "possible is %d)\n", MAX_MEM_REGIONS);
                    ret = -ENOMEM;
                    goto out;
                }

                void* mem_region_addr = *((void**)sub_regions[i].encl_addr);
                if (!mem_region_addr)
                    continue;

                mem_regions[mem_regions_cnt].toml_array = sub_regions[i].mem_ptr;
                mem_regions[mem_regions_cnt].encl_addr  = mem_region_addr;
                mem_regions[mem_regions_cnt].adjacent   = k > 0;
                mem_regions_cnt++;
            }

            sub_regions[i].size = sizeof(void*); /* rewire to actual size of sub-region */
        }
    }

    *_sub_regions_cnt = sub_regions_cnt;
    ret = 0;
out:
#if 0
    for (int i = 0; i < sub_regions_cnt; i++) {
        /* we print 4B memory cells in hope that this memory cell is accessible (only for debug) */
        uint32_t value = sub_regions[i].encl_addr ? *((uint32_t*)sub_regions[i].encl_addr) : 0;
        log_error("===== sub_region[%d]: \'%s\', size=%ld, addr=%p, value=%u\n",
                i, sub_regions[i].name, sub_regions[i].size, sub_regions[i].encl_addr, value);
    }
#endif
    for (int i = 0; i < sub_regions_cnt; i++) {
#if 0
        if (ret == 0) {
            /* misc sanity checks on all collected sub regions */
            if (sub_regions[i].size < 0) {
                log_error("Invalid deep-copy syntax (\'size\' is unspecified/negative in sub "
                        "region \'%s\')\n", sub_regions[i].name);
                ret = -EINVAL;
            }

            if (sub_regions[i].size > 0 && (!sub_regions[i].encl_addr ||
                    sub_regions[i].encl_addr == (void*)-1)) {
                /* enclave address is invalid, user provided bad struct */
                ret = -EFAULT;
            }
        }
#endif

        /* "name" fields are not needed after we collected all sub_regions */
        if (sub_regions[i].name_need_free)
            free(sub_regions[i].name);
        if (sub_regions[i].size_name_need_free)
            free(sub_regions[i].size_name);
        sub_regions[i].name = NULL;
        sub_regions[i].size_name = NULL;
    }
    return ret;
}

static void copy_sub_regions_to_untrusted(struct sub_region* sub_regions, int sub_regions_cnt,
                                         void* untrusted_addr) {
    char* cur_untrusted_addr = untrusted_addr; /* char* type for pointer arithmetic */
    for (int i = 0; i < sub_regions_cnt; i++) {
        if (sub_regions[i].size <= 0 || !sub_regions[i].encl_addr)
            continue;

        if (sub_regions[i].align > 0) {
            char* aligned_untrusted_addr = ALIGN_UP_PTR(cur_untrusted_addr, sub_regions[i].align);
            memset(cur_untrusted_addr, 0, aligned_untrusted_addr - cur_untrusted_addr);
            cur_untrusted_addr = aligned_untrusted_addr;
        }

        sub_regions[i].untrusted_addr = cur_untrusted_addr;
        if (sub_regions[i].type == COPY_OUT_ENCLAVE || sub_regions[i].type == COPY_INOUT_ENCLAVE) {
            memcpy(cur_untrusted_addr, sub_regions[i].encl_addr, sub_regions[i].size);
        } else {
            memset(cur_untrusted_addr, 0, sub_regions[i].size);
        }
        cur_untrusted_addr += sub_regions[i].size;
    }

    for (int i = 0; i < sub_regions_cnt; i++) {
        if (sub_regions[i].size <= 0 || !sub_regions[i].encl_addr)
            continue;

        if (sub_regions[i].type == COPY_PTR_ENCLAVE) {
            void* encl_ptr_value = *((void**)sub_regions[i].encl_addr);
            /* rewire pointer value in untrusted memory to a corresponding untrusted sub-region */
            for (int j = 0; j < sub_regions_cnt; j++) {
                if (sub_regions[j].encl_addr == encl_ptr_value) {
                    *((void**)sub_regions[i].untrusted_addr) = sub_regions[j].untrusted_addr;
                    break;
                }
            }
        }
    }
}

static void copy_sub_regions_to_enclave(struct sub_region* sub_regions, int sub_regions_cnt) {
    for (int i = 0; i < sub_regions_cnt; i++) {
        if (sub_regions[i].size <= 0 || !sub_regions[i].encl_addr)
            continue;

        if (sub_regions[i].type == COPY_IN_ENCLAVE || sub_regions[i].type == COPY_INOUT_ENCLAVE)
            memcpy(sub_regions[i].encl_addr, sub_regions[i].untrusted_addr, sub_regions[i].size);
    }
}

static int get_ioctl_struct(unsigned int cmd, toml_array_t** out_toml_ioctl_struct) {
    int ret;

    assert(get_tcb_trts()->ioctl_scratch_space);
    char* cptr = (char*)get_tcb_trts()->ioctl_scratch_space +
                 MAX_SUB_REGIONS * sizeof(struct sub_region);
    struct cached_ioctl_struct* ioctl_struct_cache = (struct cached_ioctl_struct*)cptr;

    for (size_t i = 0; i < IOCTL_STRUCT_CACHE_SIZE; i++) {
        if (!ioctl_struct_cache[i].cmd) {
            /* the rest of cache is uninitialized, no sense in checking it */
            break;
        }

        if (ioctl_struct_cache[i].cmd == cmd) {
            assert(ioctl_struct_cache[i].toml_array);
            *out_toml_ioctl_struct = ioctl_struct_cache[i].toml_array;
            return 0;
        }
    }

    /* this IOCTL is not in the cache, must find it in the manifest and save in cache (if found) */
    toml_table_t* manifest_sgx = toml_table_in(g_pal_public_state.manifest_root, "sgx");
    if (!manifest_sgx)
        return -PAL_ERROR_NOTIMPLEMENTED;

    toml_table_t* toml_allowed_ioctls = toml_table_in(manifest_sgx, "allowed_ioctls");
    if (!toml_allowed_ioctls)
        return -PAL_ERROR_NOTIMPLEMENTED;

    ssize_t toml_allowed_ioctls_cnt = toml_table_ntab(toml_allowed_ioctls);
    if (toml_allowed_ioctls_cnt <= 0)
        return -PAL_ERROR_NOTIMPLEMENTED;

    for (ssize_t idx = 0; idx < toml_allowed_ioctls_cnt; idx++) {
        const char* toml_allowed_ioctl_key = toml_key_in(toml_allowed_ioctls, idx);
        assert(toml_allowed_ioctl_key);

        toml_table_t* toml_ioctl_table = toml_table_in(toml_allowed_ioctls, toml_allowed_ioctl_key);
        if (!toml_ioctl_table)
            continue;

        toml_raw_t toml_ioctl_request_raw = toml_raw_in(toml_ioctl_table, "request");
        if (!toml_ioctl_request_raw)
            continue;

        int64_t ioctl_request = 0x0;
        ret = toml_rtoi(toml_ioctl_request_raw, &ioctl_request);
        if (ret < 0 || ioctl_request == 0x0) {
            log_error("Invalid request value of allowed ioctl \'%s\' in manifest\n",
                    toml_allowed_ioctl_key);
            continue;
        }

        if (ioctl_request == (int64_t)cmd) {
            /* found this IOCTL request in the manifest, now must find the corresponding struct */
            toml_raw_t toml_ioctl_struct_raw = toml_raw_in(toml_ioctl_table, "struct");
            if (!toml_ioctl_struct_raw) {
                log_error("Cannot find struct value of allowed ioctl \'%s\' in manifest\n",
                        toml_allowed_ioctl_key);
                return -PAL_ERROR_NOTIMPLEMENTED;
            }

            char* ioctl_struct_str = NULL;
            ret = toml_rtos(toml_ioctl_struct_raw, &ioctl_struct_str);
            if (ret < 0) {
                log_error("Invalid struct value of allowed ioctl \'%s\' in manifest "
                               "(sgx.allowed_ioctls.[identifier].struct must be a TOML string)\n",
                               toml_allowed_ioctl_key);
                return -PAL_ERROR_INVAL;
            }

            toml_table_t* toml_ioctl_structs = toml_table_in(manifest_sgx, "ioctl_structs");
            if (!toml_ioctl_structs) {
                log_error("There are no ioctl structs found in manifest\n");
                free(ioctl_struct_str);
                return -PAL_ERROR_INVAL;
            }

            toml_array_t* toml_ioctl_struct = toml_array_in(toml_ioctl_structs, ioctl_struct_str);
            if (!toml_ioctl_struct) {
                log_error("Cannot find struct value \'%s\' of allowed ioctl \'%s\' in "
                               "manifest (or it is not a correctly formatted TOML array)\n",
                               ioctl_struct_str, toml_allowed_ioctl_key);
                free(ioctl_struct_str);
                return -PAL_ERROR_INVAL;
            }
            free(ioctl_struct_str);

            /* found the IOCTL and its struct, add to cache; note that we don't evict old IOCTLs
             * if the cache is full -- we assume that the size of the cache is enough to keep
             * all IOCTL descriptions used by the application's thread */
            for (size_t i = 0; i < IOCTL_STRUCT_CACHE_SIZE; i++) {
                if (!ioctl_struct_cache[i].cmd) {
                    /* found empty slot, add here */
                    assert(!ioctl_struct_cache[i].toml_array);
                    ioctl_struct_cache[i].cmd        = cmd;
                    ioctl_struct_cache[i].toml_array = toml_ioctl_struct;
                    break;
                }
            }

            *out_toml_ioctl_struct = toml_ioctl_struct;
            return 0;
        }
    }

    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* Thread-local scratch space for IOCTL internal data:
 *   1. Subregions array of size MAX_SUB_REGIONS +
 *   2. Ioctl-struct cache of size IOCTL_STRUCT_CACHE_SIZE +
 *   3. Subregions cache of size SUB_REGION_INFO_CACHE_SIZE
 *
 * Note that this scratch space is allocated once per thread and never freed. Also, we assume that
 * IOCTLs during signal handling are impossible, so there is no need to protect via atomic variable
 * like `ocall_mmap_untrusted_cache: in_use`. */
static int init_ioctl_scratch_space(void) {
    if (!get_tcb_trts()->ioctl_scratch_space) {
        size_t total_size = 0;
        total_size += MAX_SUB_REGIONS * sizeof(struct sub_region);
        total_size += IOCTL_STRUCT_CACHE_SIZE * sizeof(struct cached_ioctl_struct);
        total_size += SUB_REGION_INFO_CACHE_SIZE * sizeof(struct cached_sub_region_info);
        void* scratch_space = calloc(1, total_size);
        if (!scratch_space)
            return -PAL_ERROR_NOMEM;
        get_tcb_trts()->ioctl_scratch_space = scratch_space;
    }
    return 0;
}

int _DkDeviceIoControl(PAL_HANDLE handle, unsigned int cmd, uint64_t arg) {
    int ret;
    if(handle != NULL) {
        switch (PAL_GET_TYPE(handle)) {
            case PAL_TYPE_TCP:
	    case PAL_TYPE_TCPSRV:
            case PAL_TYPE_UDP:
            case PAL_TYPE_UDPSRV: {
                break;
            }
            default:
                return -PAL_ERROR_INVAL;
        }
    }

    ret = init_ioctl_scratch_space();
    if (ret < 0)
        return ret;

    toml_array_t* toml_ioctl_struct = NULL;
    ret = get_ioctl_struct(cmd, &toml_ioctl_struct);
    if (ret < 0)
        return ret;

    assert(toml_ioctl_struct);
    if (toml_array_nelem(toml_ioctl_struct) == 0) {
        /* special case of an empty TOML array == base-type or ignored IOCTL argument */
        ret = ocall_ioctl(handle?handle->sock.fd:0, cmd, arg);
        return ret < 0 ? unix_to_pal_error(ret) : 0;
    }

    int sub_regions_cnt = MAX_SUB_REGIONS;
    struct sub_region* sub_regions = (struct sub_region*)get_tcb_trts()->ioctl_scratch_space;
    assert(sub_regions);

    /* typical IOCTL case: deep-copy the IOCTL argument's input data outside of enclave, execute the
     * IOCTL OCALL, and deep-copy the IOCTL argument's output data back into enclave */
    ret = collect_sub_regions(toml_ioctl_struct, (void*)arg, sub_regions, &sub_regions_cnt);
    if (ret < 0) {
        if (ret != -EFAULT) {
            log_error("Invalid struct format of allowed ioctl \'%u\' in manifest\n", cmd);
        }
        return unix_to_pal_error(ret);
    }

    void* untrusted_addr  = NULL;
    size_t untrusted_size = 0;
    for (int i = 0; i < sub_regions_cnt; i++)
        untrusted_size += sub_regions[i].size + sub_regions[i].align;

    bool need_munmap = false;
    ret = ocall_mmap_untrusted_cache(ALLOC_ALIGN_UP(untrusted_size), &untrusted_addr, &need_munmap);
    if (ret < 0)
        return -PAL_ERROR_NOMEM;

    copy_sub_regions_to_untrusted(sub_regions, sub_regions_cnt, untrusted_addr);

    ret = ocall_ioctl(handle?handle->sock.fd:0, cmd, (uint64_t)untrusted_addr);
    if (ret < 0) {
        ocall_munmap_untrusted_cache(untrusted_addr, ALLOC_ALIGN_UP(untrusted_size), need_munmap);
        return unix_to_pal_error(ret);
    }

    copy_sub_regions_to_enclave(sub_regions, sub_regions_cnt);

    ocall_munmap_untrusted_cache(untrusted_addr, ALLOC_ALIGN_UP(untrusted_size), need_munmap);
    return 0;
}

