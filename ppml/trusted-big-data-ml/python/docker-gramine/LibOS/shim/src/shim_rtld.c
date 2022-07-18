/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains code for loading ELF binaries in library OS. The source was originally based
 * on glibc (dl-load.c), but has been significantly modified since.
 *
 * Here is a short overview of the ELFs involved:
 *
 *  - PAL and LibOS binaries: not handled here (loaded before starting LibOS)
 *  - vDSO: loaded here
 *  - Program binary, and its interpreter (ld.so) if any: loaded here
 *  - Additional libraries: loaded by ld.so; only reported to PAL here (register_library)
 *
 * Note that we don't perform any dynamic linking here, just execute load commands and transfer
 * control to ld.so. In that regard, this file is more similar to Linux kernel (see binfmt_elf.c)
 * than glibc.
 */

#include <asm/mman.h>
#include <endian.h>
#include <errno.h>

#include "asan.h"
#include "elf.h"
#include "shim_checkpoint.h"
#include "shim_flags_conv.h"
#include "shim_fs.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_utils.h"
#include "shim_vdso.h"
#include "shim_vdso-arch.h"
#include "shim_vma.h"

#define INTERP_PATH_SIZE 256 /* Default shebang size */

/*
 * Structure describing a loaded ELF object. Originally based on glibc link_map structure.
 */
struct link_map {
    /*
     * Difference between virtual addresses (p_vaddr) in ELF file and actual virtual addresses in
     * memory. Equal to 0x0 for ET_EXECs.
     *
     * Note that `l_base_diff` name is a departure from the ELF standard and Glibc code: the ELF
     * standard uses the term "Base Address" and Glibc uses `l_addr`. We find ELF/Glibc terms
     * misleading and therefore replace them with `l_base_diff`. For the context, below is the
     * exerpt from the ELF spec, Section "Program Header", Sub-section "Base Address":
     *
     *   The virtual addresses in the program headers might not represent the actual virtual
     *   addresses of the program's memory image. [...] The difference between the virtual address
     *   of any segment in memory and the corresponding virtual address in the file is thus a single
     *   constant value for any one executable or shared object in a given process. This difference
     *   is the *base address*. One use of the base address is to relocate the memory image of the
     *   program during dynamic linking.
     */
    elf_addr_t l_base_diff;

    /* Object identifier: file path, or PAL URI if path is unavailable. */
    const char* l_name;

    /* Pointer to program header table. */
    elf_phdr_t* l_phdr;

    /* Entry point location. */
    elf_addr_t l_entry;

    /* Number of program header entries.  */
    elf_half_t l_phnum;

    /* Start and finish of memory map for this object. */
    elf_addr_t l_map_start, l_map_end;

    const char* l_interp_libname;

    /* Pointer to related file. */
    struct shim_handle* l_file;

    /* Size of all the data segments (including BSS), for setting up the brk region */
    size_t l_data_segment_size;
};

struct loadcmd {
    /*
     * Load command for a single segment. The following properties are true:
     *
     *   - start <= data_end <= map_end <= alloc_end
     *   - start, map_end, alloc_end are page-aligned
     *   - map_off is page-aligned
     *
     * The addresses are not relocated (i.e. you need to add l_base_diff to them).
     */

    /* Start of memory area */
    elf_addr_t start;

    /* End of file data (data_end .. alloc_end should be zeroed out) */
    elf_addr_t data_end;

    /* End of mapped file data (data_end rounded up to page size, so that we can mmap
     * start .. map_end) */
    elf_addr_t map_end;

    /* End of memory area */
    elf_addr_t alloc_end;

    /* Offset from the beginning of file at which the first byte of the segment resides */
    elf_off_t map_off;

    /* Permissions for memory area */
    int prot;
};

static struct link_map* g_exec_map = NULL;
static struct link_map* g_interp_map = NULL;

static int read_file_fragment(struct shim_handle* file, void* buf, size_t size, file_off_t offset);

static struct link_map* new_elf_object(const char* realname) {
    struct link_map* new;

    new = (struct link_map*)malloc(sizeof(struct link_map));
    if (new == NULL)
        return NULL;

    /* We apparently expect this to be zeroed. */
    memset(new, 0, sizeof(struct link_map));
    new->l_name = realname;

    return new;
}

static int read_loadcmd(const elf_phdr_t* ph, struct loadcmd* c) {
    assert(ph->p_type == PT_LOAD);

    if (ph->p_align > 1) {
        if (!IS_POWER_OF_2(ph->p_align)) {
            log_debug("%s: ELF load command alignment value is not a power of 2", __func__);
            return -EINVAL;
        }
        if (!IS_ALIGNED_POW2(ph->p_vaddr - ph->p_offset, ph->p_align)) {
            log_debug("%s: ELF load command address/offset not properly aligned", __func__);
            return -EINVAL;
        }
    }

    if (!IS_ALLOC_ALIGNED(ph->p_vaddr - ph->p_offset)) {
        log_debug("%s: ELF load command address/offset not page-aligned", __func__);
        return -EINVAL;
    }

    if (ph->p_filesz > ph->p_memsz) {
        log_debug("%s: file size larger than memory size", __func__);
        return -EINVAL;
    }

    c->start = ALLOC_ALIGN_DOWN(ph->p_vaddr);
    c->data_end = ph->p_vaddr + ph->p_filesz;
    c->map_end = ALLOC_ALIGN_UP(ph->p_vaddr + ph->p_filesz);
    c->alloc_end  = ALLOC_ALIGN_UP(ph->p_vaddr + ph->p_memsz);
    c->map_off = ALLOC_ALIGN_DOWN(ph->p_offset);
    assert(c->start <= c->data_end);
    assert(c->data_end <= c->map_end);
    assert(c->map_end <= c->alloc_end);

    c->prot = (((ph->p_flags & PF_R) ? PROT_READ : 0) |
               ((ph->p_flags & PF_W) ? PROT_WRITE : 0) |
               ((ph->p_flags & PF_X) ? PROT_EXEC : 0));

    return 0;
}

static int read_all_loadcmds(const elf_phdr_t* phdr, size_t phnum, size_t* n_loadcmds,
                             struct loadcmd** loadcmds) {
    const elf_phdr_t* ph;
    int ret;

    size_t n = 0;
    for (ph = phdr; ph < &phdr[phnum]; ph++)
        if (ph->p_type == PT_LOAD)
            n++;

    if (n == 0) {
        *n_loadcmds = 0;
        *loadcmds = NULL;
        return 0;
    }

    if ((*loadcmds = malloc(n * sizeof(**loadcmds))) == NULL) {
        log_debug("%s: failed to allocate memory", __func__);
        return -ENOMEM;
    }

    struct loadcmd* c = *loadcmds;
    const elf_phdr_t* ph_prev = NULL;
    for (ph = phdr; ph < &phdr[phnum]; ph++) {
        if (ph->p_type == PT_LOAD) {
            if (ph_prev && !(ph_prev->p_vaddr < ph->p_vaddr)) {
                log_debug("%s: PT_LOAD segments are not in ascending order", __func__);
                ret = -EINVAL;
                goto err;
            }

            ph_prev = ph;

            if ((ret = read_loadcmd(ph, c)) < 0)
                goto err;

            c++;
        }
    }

    *n_loadcmds = n;
    return 0;

err:
    *n_loadcmds = 0;
    free(*loadcmds);
    *loadcmds = NULL;
    return ret;
}

/*
 * Find an initial memory area for a shared object. This bookkeeps the area to make sure we can
 * access all of it, but doesn't actually map the memory: we will do that when loading the segments.
 */
static int reserve_dyn(size_t total_size, void** addr) {
    int ret;

    if ((ret = bkeep_mmap_any_aslr(ALLOC_ALIGN_UP(total_size), PROT_NONE, VMA_UNMAPPED,
                                   /*file=*/NULL, /*offset=*/0, /*comment=*/NULL, addr) < 0)) {
        log_debug("reserve_dyn: failed to find an address for shared object");
        return ret;
    }

    return 0;
}

/*
 * Execute a single load command: bookkeep the memory, map the file content, and make sure the area
 * not mapped to a file (ph_filesz .. ph_memsz) is zero-filled.
 *
 * This function doesn't undo allocations in case of error: if it fails, it may leave some segments
 * already allocated.
 */
static int execute_loadcmd(const struct loadcmd* c, elf_addr_t base_diff,
                           struct shim_handle* file) {
    int ret;
    int map_flags = MAP_FIXED | MAP_PRIVATE;
    pal_prot_flags_t pal_prot = LINUX_PROT_TO_PAL(c->prot, map_flags);

    /* Map the part that should be loaded from file, rounded up to page size. */
    if (c->start < c->map_end) {
        void* map_start = (void*)(c->start + base_diff);
        size_t map_size = c->map_end - c->start;

        if ((ret = bkeep_mmap_fixed(map_start, map_size, c->prot, map_flags, file, c->map_off,
                                    /*comment=*/NULL)) < 0) {
            log_debug("%s: failed to bookkeep address of segment", __func__);
            return ret;
        }

        if ((ret = file->fs->fs_ops->mmap(file, map_start, map_size, c->prot, map_flags,
                                          c->map_off)) < 0) {
            log_debug("%s: failed to map segment: %d", __func__, ret);
            return ret;
        }
    }

    /* Zero out the extra data at the end of mapped area. If necessary, temporarily remap the last
     * page as writable. */
    if (c->data_end < c->map_end) {
        void* zero_start = (void*)(c->data_end + base_diff);
        size_t zero_size = c->map_end - c->data_end;
        void* last_page_start = ALLOC_ALIGN_DOWN_PTR(zero_start);

        if ((c->prot & PROT_WRITE) == 0) {
            if ((ret = DkVirtualMemoryProtect(last_page_start, ALLOC_ALIGNMENT,
                                              pal_prot | PAL_PROT_WRITE) < 0)) {
                log_debug("%s: cannot change memory protections", __func__);
                return pal_to_unix_errno(ret);
            }
        }

        memset(zero_start, 0, zero_size);

        if ((c->prot & PROT_WRITE) == 0) {
            if ((ret = DkVirtualMemoryProtect(last_page_start, ALLOC_ALIGNMENT, pal_prot) < 0)) {
                log_debug("%s: cannot change memory protections", __func__);
                return pal_to_unix_errno(ret);
            }
        }
    }

    /* Allocate extra pages after the mapped area. */
    if (c->map_end < c->alloc_end) {
        void* zero_page_start = (void*)(c->map_end + base_diff);
        size_t zero_page_size = c->alloc_end - c->map_end;
        int zero_map_flags = MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS;
        pal_prot_flags_t zero_pal_prot = LINUX_PROT_TO_PAL(c->prot, zero_map_flags);

        if ((ret = bkeep_mmap_fixed(zero_page_start, zero_page_size, c->prot, zero_map_flags,
                                    /*file=*/NULL, /*offset=*/0, /*comment=*/NULL)) < 0) {
            log_debug("%s: cannot bookkeep address of zero-fill pages", __func__);
            return ret;
        }

        if ((ret = DkVirtualMemoryAlloc(&zero_page_start, zero_page_size, /*alloc_type=*/0,
                                        zero_pal_prot)) < 0) {
            log_debug("%s: cannot map zero-fill pages", __func__);
            return pal_to_unix_errno(ret);
        }
    }

    return 0;
}

static struct link_map* map_elf_object(struct shim_handle* file, elf_ehdr_t* ehdr) {
    elf_phdr_t* phdr = NULL;
    elf_addr_t interp_libname_vaddr = 0;
    struct loadcmd* loadcmds = NULL;
    size_t n_loadcmds = 0;
    const char* errstring = NULL;
    int ret = 0;

    /* Check if the file is valid. */

    if (!(file && file->fs && file->fs->fs_ops))
        return NULL;

    if (!(file->fs->fs_ops->read && file->fs->fs_ops->mmap && file->fs->fs_ops->seek))
        return NULL;

    /* Allocate a new link_map. */

    const char* name = file->uri;
    struct link_map* l = new_elf_object(name);

    if (!l)
        return NULL;

    /* Load the program header table. */

    size_t phdr_size = ehdr->e_phnum * sizeof(elf_phdr_t);
    phdr = (elf_phdr_t*)malloc(phdr_size);
    if (!phdr) {
        errstring = "phdr malloc failure";
        ret = -ENOMEM;
        goto err;
    }
    if ((ret = read_file_fragment(file, phdr, phdr_size, ehdr->e_phoff)) < 0) {
        errstring = "cannot read phdr";
        goto err;
    }

    /* Scan the program header table load commands and additional information. */

    if ((ret = read_all_loadcmds(phdr, ehdr->e_phnum, &n_loadcmds, &loadcmds)) < 0) {
        errstring = "failed to read load commands";
        goto err;
    }

    if (n_loadcmds == 0) {
        /* This only happens for a malformed object, and the calculations below assume the loadcmds
         * array is not empty. */
        errstring = "object file has no loadable segments";
        ret = -EINVAL;
        goto err;
    }

    const elf_phdr_t* ph;
    for (ph = phdr; ph < &phdr[ehdr->e_phnum]; ph++) {
        if (ph->p_type == PT_INTERP) {
            interp_libname_vaddr = ph->p_vaddr;
        }
    }

    /* Determine the load address. */

    size_t load_start = loadcmds[0].start;
    size_t load_end = loadcmds[n_loadcmds - 1].alloc_end;

    if (ehdr->e_type == ET_DYN) {
        /*
         * This is a position-independent shared object, reserve a memory area to determine load
         * address.
         *
         * Note that we reserve memory starting from offset 0, not from load_start. This is to
         * ensure that l_base_diff will not be less than 0.
         */
        void* addr;

        if ((ret = reserve_dyn(load_end, &addr)) < 0) {
            errstring = "failed to allocate memory for shared object";
            goto err;
        }

        l->l_base_diff = (elf_addr_t)addr;
    } else {
        /* This is a non-pie shared object (EXEC, executable), so the difference between virtual
         * addresses (p_vaddr) in the ELF file and the actual addresses in memory is zero. */
        l->l_base_diff = 0x0;
    }
    l->l_map_start = load_start + l->l_base_diff;
    l->l_map_end   = load_end + l->l_base_diff;

    /* Execute load commands. */
    l->l_data_segment_size = 0;
    for (struct loadcmd* c = &loadcmds[0]; c < &loadcmds[n_loadcmds]; c++) {
        if ((ret = execute_loadcmd(c, l->l_base_diff, file)) < 0) {
            errstring = "failed to execute load command";
            goto err;
        }

        if (!l->l_phdr && ehdr->e_phoff >= c->map_off
                && ehdr->e_phoff + phdr_size <= c->map_off + (c->data_end - c->start)) {
            /* Found the program header in this segment. */
            elf_addr_t phdr_vaddr = ehdr->e_phoff - c->map_off + c->start;
            l->l_phdr = (elf_phdr_t*)(phdr_vaddr + l->l_base_diff);
        }

        if (interp_libname_vaddr != 0 && !l->l_interp_libname && c->start <= interp_libname_vaddr
                && interp_libname_vaddr < c->data_end) {
            /* Found the interpreter name in this segment (but we need to validate length). */
            const char* interp_libname = (const char*)(interp_libname_vaddr + l->l_base_diff);
            size_t maxlen = c->data_end - interp_libname_vaddr;
            size_t len = strnlen(interp_libname, maxlen);
            if (len == maxlen) {
                errstring = "interpreter name is longer than mapped segment";
                ret = -EINVAL;
                goto err;
            }
            l->l_interp_libname = interp_libname;
        }

        if (ehdr->e_entry != 0 && !l->l_entry && c->start <= ehdr->e_entry
                && ehdr->e_entry < c->data_end) {
            /* Found the entry point in this segment. */
            l->l_entry = (elf_addr_t)(ehdr->e_entry + l->l_base_diff);
        }

        if (!(c->prot & PROT_EXEC))
            l->l_data_segment_size += c->alloc_end - c->start;
    }

    /* Check if various fields were found in mapped segments (if specified at all). */

    if (!l->l_phdr) {
        errstring = "program header not found in any of the segments";
        ret = -EINVAL;
        goto err;
    }

    if (interp_libname_vaddr != 0 && !l->l_interp_libname) {
        errstring = "interpreter name not found in any of the segments";
        ret = -EINVAL;
        goto err;
    }

    if (ehdr->e_entry != 0 && !l->l_entry) {
        errstring = "entry point not found in any of the segments";
        ret = -EINVAL;
        goto err;
    }

    /* Fill in remaining link_map information. */

    l->l_phnum = ehdr->e_phnum;

    free(phdr);
    free(loadcmds);
    return l;

err:
    log_debug("loading %s: %s (%d)", l->l_name, errstring, ret);
    free(phdr);
    free(loadcmds);
    free(l);
    return NULL;
}

static void remove_elf_object(struct link_map* l) {
    remove_r_debug((void*)l->l_base_diff);
    free(l);
}

static int check_elf_header(elf_ehdr_t* ehdr) {
    const char* errstring __attribute__((unused));

#if __BYTE_ORDER == __BIG_ENDIAN
#define byteorder  ELFDATA2MSB
#elif __BYTE_ORDER == __LITTLE_ENDIAN
#define byteorder ELFDATA2LSB
#else
#error "Unknown __BYTE_ORDER " __BYTE_ORDER
#define byteorder ELFDATANONE
#endif

    static const unsigned char expected[EI_NIDENT] = {
        [EI_MAG0] = ELFMAG0,       [EI_MAG1] = ELFMAG1,           [EI_MAG2] = ELFMAG2,
        [EI_MAG3] = ELFMAG3,       [EI_CLASS] = ELF_NATIVE_CLASS, [EI_DATA] = byteorder,
        [EI_VERSION] = EV_CURRENT, [EI_OSABI] = 0,
    };

#undef byteorder

    /* See whether the ELF header is what we expect.  */
    if (memcmp(ehdr->e_ident, expected, EI_OSABI) != 0 ||
            (ehdr->e_ident[EI_OSABI] != ELFOSABI_SYSV &&
             ehdr->e_ident[EI_OSABI] != ELFOSABI_LINUX)) {
        errstring = "ELF file with invalid header";
        goto verify_failed;
    }

    if (memcmp(&ehdr->e_ident[EI_PAD], &expected[EI_PAD], EI_NIDENT - EI_PAD) != 0) {
        errstring = "nonzero padding in e_ident";
        goto verify_failed;
    }

    /* Now we check if the host match the elf machine profile */
    if (ehdr->e_machine != SHIM_ELF_HOST_MACHINE) {
        errstring = "ELF file does not match with the host";
        goto verify_failed;
    }

    /* check if the type of ELF header is either DYN or EXEC */
    if (ehdr->e_type != ET_DYN && ehdr->e_type != ET_EXEC) {
        errstring = "only ET_DYN and ET_EXEC can be loaded";
        goto verify_failed;
    }

    /* check if phentsize match the size of elf_phdr_t */
    if (ehdr->e_phentsize != sizeof(elf_phdr_t)) {
        errstring = "ELF file's phentsize has unexpected size";
        goto verify_failed;
    }

    return 0;

verify_failed:
    log_debug("loading ELF file failed: %s", errstring);
    return -EINVAL;
}

static int read_partial_fragment(struct shim_handle* file, void* buf, size_t size,
                                 file_off_t offset, size_t* out_bytes_read) {
    if (!file)
        return -EINVAL;

    if (!file->fs || !file->fs->fs_ops || !file->fs->fs_ops->read)
        return -EACCES;

    ssize_t pos = offset;
    ssize_t read_ret = file->fs->fs_ops->read(file, buf, size, &pos);

    if (read_ret < 0)
        return read_ret;

    *out_bytes_read = (size_t)read_ret;
    return 0;
}

static int read_file_fragment(struct shim_handle* file, void* buf, size_t size, file_off_t offset) {
    size_t read_size = 0;
    int ret = read_partial_fragment(file, buf, size, offset, &read_size);
    if (ret < 0) {
        return ret;
    }

    if (read_size < size) {
        return -EINVAL;
    }
    return 0;
}

/* Note that `**out_new_argv` is allocated as a single object -- a concatenation of all argv
 * strings; caller of this function should do a single free(**out_new_argv). */
static int load_and_check_shebang(struct shim_handle* file, char** argv,
                                  char*** out_new_argv) {
    int ret;

    char** new_argv = NULL;
    char shebang[INTERP_PATH_SIZE];

    size_t shebang_len = 0;
    ret = read_partial_fragment(file, shebang, sizeof(shebang), /*offset=*/0, &shebang_len);
    if (ret < 0 || shebang_len < 2 || shebang[0] != '#' || shebang[1] != '!') {
        char* path = NULL;
        if (file->dentry) {
            /* this may fail, but we are already inside a more serious error handler */
            dentry_abs_path(file->dentry, &path, /*size=*/NULL);
        }
        log_error("Failed to read shebang line from %s", path ? path : "(unknown)");
        free(path);
        return -ENOEXEC;
    }

    /* Strip shebang starting sequence */
    shebang_len -= 2;
    memmove(&shebang[0], &shebang[2], shebang_len);
    shebang[shebang_len] = 0;

    /* Strip extra space characters */
    for (char* p = shebang; *p; p++) {
        if (*p != ' ') {
            shebang_len -= p - shebang;
            memmove(shebang, p, shebang_len);
            break;
        }
    }
    shebang[shebang_len] = 0;

    /* Strip new line character */
    char* newlineptr = strchr(shebang, '\n');
    if (newlineptr)
        *newlineptr = 0;

    char* interp = shebang;

    /* Separate args and interp path */
    char* spaceptr = strchr(interp, ' ');
    if (spaceptr)
        *spaceptr = 0;

    const char* argv_shebang[] = {interp, spaceptr ? spaceptr + 1 : NULL, NULL};

    size_t new_argv_bytes = 0, new_argv_cnt = 0;
    for (const char** a = argv_shebang; *a; a++) {
        new_argv_bytes += strlen(*a) + 1;
        new_argv_cnt++;
    }
    for (char** a = argv; *a; a++) {
        new_argv_bytes += strlen(*a) + 1;
        new_argv_cnt++;
    }

    log_debug("Assembling %zu execve arguments (total size is %zu bytes)", new_argv_cnt,
              new_argv_bytes);

    new_argv = calloc(new_argv_cnt + 1, sizeof(*new_argv));
    if (!new_argv) {
        ret = -ENOMEM;
        goto err;
    }

    char* new_argv_ptr = malloc(new_argv_bytes);
    if (!new_argv_ptr) {
        ret = -ENOMEM;
        goto err;
    }

    size_t new_argv_idx = 0;
    for (const char** a = argv_shebang; *a; a++) {
        size_t size = strlen(*a) + 1;
        memcpy(new_argv_ptr, *a, size);
        new_argv[new_argv_idx] = new_argv_ptr;
        new_argv_idx++;
        new_argv_ptr += size;
    }
    for (char** a = argv; *a; a++) {
        size_t size = strlen(*a) + 1;
        memcpy(new_argv_ptr, *a, size);
        new_argv[new_argv_idx] = new_argv_ptr;
        new_argv_idx++;
        new_argv_ptr += size;
    }
    new_argv[new_argv_idx] = NULL;

    log_debug("Interpreter to be used for execve: %s", *new_argv);
    *out_new_argv = new_argv;
    return 0;

err:
    if (new_argv) {
        free(*new_argv);
        free(new_argv);
    }
    return ret;
}

/* Note that `**out_new_argv` is allocated as a single object -- a concatenation of all argv
 * strings; caller of this function should do a single free(**out_new_argv). */
int load_and_check_exec(const char* path, const char** argv, struct shim_handle** out_exec,
                        char*** out_new_argv) {
    int ret;

    struct shim_handle* file = NULL;

    /* immediately copy `argv` into `curr_argv`; this simplifies ownership tracking because this way
     * `*out_new_argv` must be always freed by caller */
    size_t curr_argv_bytes = 0, curr_argv_cnt = 0;
    for (const char** a = argv; *a; a++) {
        curr_argv_bytes += strlen(*a) + 1;
        curr_argv_cnt++;
    }

    char** curr_argv = calloc(curr_argv_cnt + 1, sizeof(*curr_argv));
    if (!curr_argv) {
        ret = -ENOMEM;
        goto err;
    }

    char* curr_argv_ptr = malloc(curr_argv_bytes);
    if (!curr_argv_ptr) {
        ret = -ENOMEM;
        goto err;
    }

    size_t curr_argv_idx = 0;
    for (const char** a = argv; *a; a++) {
        size_t size = strlen(*a) + 1;
        memcpy(curr_argv_ptr, *a, size);
        curr_argv[curr_argv_idx] = curr_argv_ptr;
        curr_argv_idx++;
        curr_argv_ptr += size;
    }
    curr_argv[curr_argv_idx] = NULL;

    if (curr_argv_idx == 0) {
        /* tricky corner case: if argv is empty, then `free(*curr_argv)` is a no-op */
        free(curr_argv_ptr);
    }

    size_t depth = 0;
    while (true) {
        if (depth++ > 5) {
            ret = -ELOOP;
            goto err;
        }

        file = get_new_handle();
        if (!file) {
            ret = -ENOMEM;
            goto err;
        }

        ret = open_executable(file, depth > 1 ? curr_argv[0] : path);
        if (ret < 0) {
            goto err;
        }

        ret = check_elf_object(file);
        if (ret == 0) {
            /* Success */
            break;
        }

        log_debug("File %s not recognized as ELF, looking for shebang",
                  depth > 1 ? curr_argv[0] : path);

        char** new_argv = NULL;
        ret = load_and_check_shebang(file, curr_argv, &new_argv);
        if (ret < 0) {
            goto err;
        }

        /* clean up after previous iteration */
        free(*curr_argv);
        free(curr_argv);
        put_handle(file);

        curr_argv = new_argv;
    }

    *out_exec = file;
    *out_new_argv = curr_argv;
    return 0;

err:
    if (file)
        put_handle(file);
    if (curr_argv) {
        free(*curr_argv);
        free(curr_argv);
    }
    return ret;
}

static int load_elf_header(struct shim_handle* file, elf_ehdr_t* ehdr) {
    int ret = read_file_fragment(file, ehdr, sizeof(*ehdr), /*offset=*/0);
    if (ret < 0) {
        return -ENOEXEC;
    }

    ret = check_elf_header(ehdr);
    if (ret < 0) {
        return -ENOEXEC;
    }

    return 0;
}

int check_elf_object(struct shim_handle* file) {
    elf_ehdr_t ehdr;
    return load_elf_header(file, &ehdr);
}

int load_elf_object(struct shim_handle* file, struct link_map** out_map) {
    int ret;
    assert(file);

    const char* fname = file->uri;
    log_debug("loading \"%s\"", fname);

    elf_ehdr_t ehdr;
    if ((ret = load_elf_header(file, &ehdr)) < 0)
        return ret;

    struct link_map* map = map_elf_object(file, &ehdr);
    if (!map) {
        log_error("Failed to map %s. This may be caused by the binary being non-PIE, in which "
                  "case Gramine requires a specially-crafted memory layout. You can enable it "
                  "by adding 'sgx.nonpie_binary = true' to the manifest.",
                  fname);
        return -EINVAL;
    }

    get_handle(file);
    map->l_file = file;

    if (map->l_file && map->l_file->uri) {
        append_r_debug(map->l_file->uri, (void*)map->l_base_diff);
    }

    *out_map = map;
    return 0;
}

static bool need_interp(struct link_map* exec_map) {
    return exec_map->l_interp_libname != NULL;
}

extern const char** g_library_paths;

static int find_interp(const char* interp_name, struct shim_dentry** out_dent) {
    assert(locked(&g_dcache_lock));

    size_t interp_name_len = strlen(interp_name);
    const char* filename = interp_name;
    size_t filename_len = interp_name_len;

    for (size_t i = 0; i < interp_name_len; i++) {
        if (interp_name[i] == '/') {
            filename = interp_name + i + 1;
            filename_len = interp_name_len - i - 1;
        }
    }

    const char* default_paths[] = {"/lib", "/lib64", NULL};
    const char** paths          = g_library_paths ?: default_paths;

    for (const char** path = paths; *path; path++) {
        size_t path_len = strlen(*path);
        char* interp_path = alloc_concat3(*path, path_len, "/", 1, filename, filename_len);
        if (!interp_path) {
            log_warning("%s: couldn't allocate path: %s/%s", __func__, *path, filename);
            return -ENOMEM;
        }

        log_debug("%s: searching for interpreter: %s", __func__, interp_path);
        struct shim_dentry* dent;
        int ret = path_lookupat(/*start=*/NULL, interp_path, LOOKUP_FOLLOW, &dent);
        if (ret == 0) {
            *out_dent = dent;
            return 0;
        }
    }

    return -ENOENT;
}

static int find_and_open_interp(const char* interp_name, struct shim_handle* hdl) {
    assert(locked(&g_dcache_lock));

    struct shim_dentry* dent;
    int ret = find_interp(interp_name, &dent);
    if (ret < 0)
        return ret;

    ret = dentry_open(hdl, dent, O_RDONLY);
    put_dentry(dent);
    return ret;
}

static int load_interp_object(struct link_map* exec_map) {
    assert(!g_interp_map);

    struct shim_handle* hdl = get_new_handle();
    if (!hdl)
        return -ENOMEM;

    lock(&g_dcache_lock);
    int ret = find_and_open_interp(exec_map->l_interp_libname, hdl);
    unlock(&g_dcache_lock);
    if (ret < 0)
        goto out;

    ret = load_elf_object(hdl, &g_interp_map);
out:
    put_handle(hdl);
    return ret;
}

int load_elf_interp(struct link_map* exec_map) {
    if (!g_interp_map && need_interp(exec_map))
        return load_interp_object(exec_map);

    return 0;
}

void remove_loaded_elf_objects(void) {
    if (g_exec_map) {
        remove_elf_object(g_exec_map);
        g_exec_map = NULL;
    }
    if (g_interp_map) {
        remove_elf_object(g_interp_map);
        g_interp_map = NULL;
    }
}

/*
 * libsysdb.so is loaded as shared library and load address for child may not match the one for
 * parent. Just treat vdso page as user-program data and adjust function pointers for vdso
 * functions after migration.
 */

static void* g_vdso_addr __attribute_migratable = NULL;

static int vdso_map_init(void) {
    /*
     * Allocate vdso page as user program allocated it.
     * Using directly vdso code in LibOS causes trouble when emulating fork.
     * In host child process, LibOS may or may not be loaded at the same address.
     * When LibOS is loaded at different address, it may overlap with the old vDSO
     * area.
     */
    void* addr = NULL;
    int ret = bkeep_mmap_any_aslr(ALLOC_ALIGN_UP(vdso_so_size), PROT_READ | PROT_EXEC,
                                  MAP_PRIVATE | MAP_ANONYMOUS, NULL, 0, LINUX_VDSO_FILENAME,
                                  &addr);
    if (ret < 0) {
        return ret;
    }

    ret = DkVirtualMemoryAlloc(&addr, ALLOC_ALIGN_UP(vdso_so_size), /*alloc_type=*/0,
                               PAL_PROT_READ | PAL_PROT_WRITE);
    if (ret < 0) {
        return pal_to_unix_errno(ret);
    }

    memcpy(addr, &vdso_so, vdso_so_size);
    memset(addr + vdso_so_size, 0, ALLOC_ALIGN_UP(vdso_so_size) - vdso_so_size);

    ret = DkVirtualMemoryProtect(addr, ALLOC_ALIGN_UP(vdso_so_size), PAL_PROT_READ | PAL_PROT_EXEC);
    if (ret < 0) {
        return pal_to_unix_errno(ret);
    }

    append_r_debug("file:[vdso_libos]", addr);
    g_vdso_addr = addr;
    return 0;
}

int init_elf_objects(void) {
    int ret = 0;

    lock(&g_process.fs_lock);
    struct shim_handle* exec = g_process.exec;
    if (exec)
        get_handle(exec);
    unlock(&g_process.fs_lock);

    if (!exec)
        return 0;

    if (!g_exec_map) {
        /* Child processes should have received `g_exec_map` from parent */
        assert(!g_pal_public_state->parent_process);

        ret = load_elf_object(exec, &g_exec_map);
        if (ret < 0)
            goto out;
    }

    ret = init_brk_from_executable(g_exec_map);
    if (ret < 0)
        goto out;

    if (!g_interp_map && need_interp(g_exec_map) && (ret = load_interp_object(g_exec_map)) < 0)
        goto out;

    ret = 0;
out:
    put_handle(exec);
    return ret;
}

int init_brk_from_executable(struct link_map* exec_map) {
    return init_brk_region((void*)ALLOC_ALIGN_UP(exec_map->l_map_end),
                           exec_map->l_data_segment_size);
}

int register_library(const char* name, unsigned long load_address) {
    log_debug("glibc register library %s loaded at 0x%08lx", name, load_address);

    struct shim_handle* hdl = get_new_handle();

    if (!hdl)
        return -ENOMEM;

    int err = open_namei(hdl, NULL, name, O_RDONLY, 0, NULL);
    if (err < 0) {
        put_handle(hdl);
        return err;
    }

    append_r_debug(hdl->uri, (void*)load_address);
    put_handle(hdl);
    return 0;
}

/*
 * Helper function for unpoisoning the stack before jump: we need to do it from a function that is
 * not instrumented by ASan, so that the stack is guaranteed to stay unpoisoned.
 *
 * Note that at this point, we could also be running from the initial PAL stack. However, we
 * unconditionally unpoison the LibOS stack for simplicity.
 */
__attribute_no_sanitize_address
noreturn static void cleanup_and_call_elf_entry(elf_addr_t entry, void* argp) {
#ifdef ASAN
    uintptr_t libos_stack_bottom = (uintptr_t)SHIM_TCB_GET(libos_stack_bottom);
    asan_unpoison_region(libos_stack_bottom - SHIM_THREAD_LIBOS_STACK_SIZE,
                         SHIM_THREAD_LIBOS_STACK_SIZE);

#endif
    call_elf_entry(entry, argp);
}

noreturn void execute_elf_object(struct link_map* exec_map, void* argp, elf_auxv_t* auxp) {
    if (exec_map) {
        /* If a new map is provided, it means we have cleared the existing one by calling
         * `remove_loaded_elf_objects`. This happens during `execve`. */
        assert(!g_exec_map);
        g_exec_map = exec_map;
    }
    assert(g_exec_map);

    int ret = vdso_map_init();
    if (ret < 0) {
        log_error("Could not initialize vDSO (error code = %d)", ret);
        process_exit(/*error_code=*/0, /*term_signal=*/SIGKILL);
    }

    /* at this point, stack looks like this:
     *
     *               +-------------------+
     *   argp +--->  |  argc             | long
     *               |  ptr to argv[0]   | char*
     *               |  ...              | char*
     *               |  NULL             | char*
     *               |  ptr to envp[0]   | char*
     *               |  ...              | char*
     *               |  NULL             | char*
     *               |  <space for auxv> |
     *               |  envp[0] string   |
     *               |  ...              |
     *               |  argv[0] string   |
     *               |  ...              |
     *               +-------------------+
     */
    assert(IS_ALIGNED_PTR(argp, 16)); /* stack must be 16B-aligned */

    static_assert(REQUIRED_ELF_AUXV >= 9, "not enough space on stack for auxv");
    auxp[0].a_type     = AT_PHDR;
    auxp[0].a_un.a_val = (__typeof(auxp[0].a_un.a_val))g_exec_map->l_phdr;
    auxp[1].a_type     = AT_PHNUM;
    auxp[1].a_un.a_val = g_exec_map->l_phnum;
    auxp[2].a_type     = AT_PAGESZ;
    auxp[2].a_un.a_val = ALLOC_ALIGNMENT;
    auxp[3].a_type     = AT_ENTRY;
    auxp[3].a_un.a_val = g_exec_map->l_entry;
    auxp[4].a_type     = AT_BASE;
    auxp[4].a_un.a_val = g_interp_map ? g_interp_map->l_map_start : 0;
    auxp[5].a_type     = AT_RANDOM;
    auxp[5].a_un.a_val = 0; /* filled later */
    auxp[6].a_type     = AT_PHENT;
    auxp[6].a_un.a_val = sizeof(elf_phdr_t);
    auxp[7].a_type     = AT_SYSINFO_EHDR;
    auxp[7].a_un.a_val = (uint64_t)g_vdso_addr;
    auxp[8].a_type     = AT_NULL;
    auxp[8].a_un.a_val = 0;

    /* populate extra memory space for aux vector data */
    static_assert(REQUIRED_ELF_AUXV_SPACE >= 16, "not enough space on stack for auxv");
    elf_addr_t auxp_extra = (elf_addr_t)&auxp[9];

    elf_addr_t random = auxp_extra; /* random 16B for AT_RANDOM */
    ret = DkRandomBitsRead((void*)random, 16);
    if (ret < 0) {
        log_error("execute_elf_object: DkRandomBitsRead failed: %d", ret);
        DkProcessExit(1);
        /* UNREACHABLE */
    }
    auxp[5].a_un.a_val = random;

    elf_addr_t entry = g_interp_map ? g_interp_map->l_entry : g_exec_map->l_entry;

    cleanup_and_call_elf_entry(entry, argp);
}

BEGIN_CP_FUNC(elf_object) {
    __UNUSED(size);
    assert(size == sizeof(struct link_map));

    struct link_map* map = (struct link_map*)obj;
    struct link_map* new_map;

    size_t off = GET_FROM_CP_MAP(obj);

    if (!off) {
        off = ADD_CP_OFFSET(sizeof(struct link_map));
        ADD_TO_CP_MAP(obj, off);

        new_map = (struct link_map*)(base + off);
        memcpy(new_map, map, sizeof(struct link_map));

        if (map->l_file)
            DO_CP_MEMBER(handle, map, new_map, l_file);

        if (map->l_name) {
            size_t namelen = strlen(map->l_name);
            char* name     = (char*)(base + ADD_CP_OFFSET(namelen + 1));
            memcpy(name, map->l_name, namelen + 1);
            new_map->l_name = name;
        }

        ADD_CP_FUNC_ENTRY(off);
    } else {
        new_map = (struct link_map*)(base + off);
    }

    if (objp)
        *objp = (void*)new_map;
}
END_CP_FUNC(elf_object)

BEGIN_RS_FUNC(elf_object) {
    __UNUSED(offset);
    struct link_map* map = (void*)(base + GET_CP_FUNC_ENTRY());

    CP_REBASE(map->l_name);
    CP_REBASE(map->l_file);
}
END_RS_FUNC(elf_object)

BEGIN_CP_FUNC(loaded_elf_objects) {
    __UNUSED(obj);
    __UNUSED(size);
    __UNUSED(objp);
    struct link_map* new_exec_map = NULL;
    struct link_map* new_interp_map = NULL;
    if (g_exec_map)
        DO_CP(elf_object, g_exec_map, &new_exec_map);
    if (g_interp_map)
        DO_CP(elf_object, g_interp_map, &new_interp_map);

    size_t off = ADD_CP_OFFSET(2 * sizeof(struct link_map*));
    struct link_map** maps = (void*)(base + off);
    maps[0] = new_exec_map;
    maps[1] = new_interp_map;

    ADD_CP_FUNC_ENTRY(off);
}
END_CP_FUNC(loaded_elf_objects)

BEGIN_RS_FUNC(loaded_elf_objects) {
    __UNUSED(base);
    __UNUSED(offset);
    struct link_map** maps = (void*)(base + GET_CP_FUNC_ENTRY());

    assert(!g_exec_map);
    g_exec_map = maps[0];
    if (g_exec_map)
        CP_REBASE(g_exec_map);

    assert(!g_interp_map);
    g_interp_map = maps[1];
    if (g_interp_map)
        CP_REBASE(g_interp_map);
}
END_RS_FUNC(loaded_elf_objects)
