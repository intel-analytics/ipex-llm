/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*!
 * This file contains entry and exit functions of library OS.
 */

#include <sys/mman.h>

#include "api.h"
#include "hex.h"
#include "init.h"
#include "pal.h"
#include "pal_error.h"
#include "shim_checkpoint.h"
#include "shim_context.h"
#include "shim_defs.h"
#include "shim_fs.h"
#include "shim_fs_lock.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_ipc.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_sync.h"
#include "shim_tcb.h"
#include "shim_thread.h"
#include "shim_utils.h"
#include "shim_vma.h"
#include "toml.h"
#include "toml_utils.h"

static_assert(sizeof(shim_tcb_t) <= PAL_LIBOS_TCB_SIZE,
              "shim_tcb_t does not fit into PAL_TCB; please increase PAL_LIBOS_TCB_SIZE");

const toml_table_t* g_manifest_root = NULL;
struct pal_public_state* g_pal_public_state = NULL;

/* This function is used by stack protector's __stack_chk_fail(), _FORTIFY_SOURCE's *_chk()
 * functions and by assert.h's assert() defined in the common library. Thus it might be called by
 * any thread, even internal. */
noreturn void shim_abort(void) {
    DEBUG_BREAK_ON_FAILURE();
    DkProcessExit(1);
}

static unsigned pal_errno_to_unix_errno_table[PAL_ERROR_NATIVE_COUNT + 1] = {
    [PAL_ERROR_SUCCESS]         = 0,
    [PAL_ERROR_NOTIMPLEMENTED]  = ENOSYS,
    [PAL_ERROR_NOTDEFINED]      = ENOSYS,
    [PAL_ERROR_NOTSUPPORT]      = EACCES,
    [PAL_ERROR_INVAL]           = EINVAL,
    [PAL_ERROR_TOOLONG]         = ENAMETOOLONG,
    [PAL_ERROR_DENIED]          = EACCES,
    [PAL_ERROR_BADHANDLE]       = EBADF,
    [PAL_ERROR_STREAMEXIST]     = EEXIST,
    [PAL_ERROR_STREAMNOTEXIST]  = ENOENT,
    [PAL_ERROR_STREAMISFILE]    = ENOTDIR,
    [PAL_ERROR_STREAMISDIR]     = EISDIR,
    [PAL_ERROR_STREAMISDEVICE]  = ESPIPE,
    [PAL_ERROR_INTERRUPTED]     = EINTR,
    [PAL_ERROR_OVERFLOW]        = EFAULT,
    [PAL_ERROR_BADADDR]         = EFAULT,
    [PAL_ERROR_NOMEM]           = ENOMEM,
    [PAL_ERROR_INCONSIST]       = EFAULT,
    [PAL_ERROR_TRYAGAIN]        = EAGAIN,
    [PAL_ERROR_NOTSERVER]       = EINVAL,
    [PAL_ERROR_NOTCONNECTION]   = ENOTCONN,
    [PAL_ERROR_CONNFAILED]      = ECONNRESET,
    [PAL_ERROR_ADDRNOTEXIST]    = EADDRNOTAVAIL,
    [PAL_ERROR_AFNOSUPPORT]     = EAFNOSUPPORT,
    [PAL_ERROR_CONNFAILED_PIPE] = EPIPE,
};

static unsigned long pal_to_unix_errno_positive(unsigned long err) {
    return (err <= PAL_ERROR_NATIVE_COUNT) ? pal_errno_to_unix_errno_table[err] : EINVAL;
}

long pal_to_unix_errno(long err) {
    if (err >= 0) {
        return pal_to_unix_errno_positive((unsigned long)err);
    }
    return -pal_to_unix_errno_positive((unsigned long)-err);
}

void* migrated_memory_start;
void* migrated_memory_end;

const char** migrated_envp __attribute_migratable;

/* `g_library_paths` is populated with LD_PRELOAD entries once during LibOS initialization and is
 * used in `__load_interp_object()` to search for ELF program interpreter in specific paths. Once
 * allocated, its memory is never freed or updated. */
char** g_library_paths = NULL;

struct shim_lock __master_lock;
bool lock_enabled;

void* allocate_stack(size_t size, size_t protect_size, bool user) {
    void* stack = NULL;

    size = ALLOC_ALIGN_UP(size);
    protect_size = ALLOC_ALIGN_UP(protect_size);

    log_debug("Allocating stack at %p (size = %ld)", stack, size);

    if (!user) {
        stack = system_malloc(size + protect_size);
        if (!stack) {
            return NULL;
        }
        stack += protect_size;
        stack = ALIGN_UP_PTR(stack, 16);
        return stack;
    }

    /* reserve non-readable non-writable page below the user stack to catch stack overflows */
    int ret = bkeep_mmap_any_aslr(size + protect_size, PROT_NONE,
                                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_GROWSDOWN, NULL, 0, "stack",
                                  &stack);
    if (ret < 0) {
        return NULL;
    }

    bool need_mem_free = false;
    ret = DkVirtualMemoryAlloc(&stack, size + protect_size, 0, /*prot=*/0);
    if (ret < 0) {
        goto out_fail;
    }
    need_mem_free = true;

    /* ensure proper alignment for process' initial stack */
    assert(IS_ALIGNED_PTR(stack, 16));

    if (bkeep_mprotect(stack + protect_size, size, PROT_READ | PROT_WRITE,
                       /*is_internal=*/false) < 0) {
        goto out_fail;
    }

    if (DkVirtualMemoryProtect(stack + protect_size, size, PAL_PROT_READ | PAL_PROT_WRITE) < 0) {
        goto out_fail;
    }

    return stack + protect_size;

out_fail:;
    void* tmp_vma = NULL;
    if (bkeep_munmap(stack, size + protect_size, /*is_internal=*/false, &tmp_vma) < 0) {
        BUG();
    }
    if (need_mem_free) {
        if (DkVirtualMemoryFree(stack, size + protect_size) < 0) {
            BUG();
        }
    }
    bkeep_remove_tmp_vma(tmp_vma);
    return NULL;
}

/* populate already-allocated stack with copied argv and envp and space for auxv;
 * returns a pointer to first stack frame (starting with argc, then argv pointers, and so on)
 * and a pointer inside first stack frame (with auxv[0], auxv[1], and so on) */
static int populate_stack(void* stack, size_t stack_size, const char** argv, const char** envp,
                          const char*** out_argp, elf_auxv_t** out_auxv) {
    void* stack_low_addr  = stack;
    void* stack_high_addr = stack + stack_size;

#define ALLOCATE_FROM_HIGH_ADDR(size)                     \
    ({                                                    \
        if ((stack_high_addr -= (size)) < stack_low_addr) \
            return -ENOMEM;                               \
        stack_high_addr;                                  \
    })

#define ALLOCATE_FROM_LOW_ADDR(size)                      \
    ({                                                    \
        if ((stack_low_addr += (size)) > stack_high_addr) \
            return -ENOMEM;                               \
        stack_low_addr - (size);                          \
    })

    /* create stack layout as follows for ld.so:
     *
     *                 +-------------------+
     * out_argp +--->  |  argc             | long
     *                 |  ptr to argv[0]   | char*
     *                 |  ptr to argv[1]   | char*
     *                 |  ...              | char*
     *                 |  NULL             | char*
     *                 |  ptr to envp[0]   | char*
     *                 |  ptr to envp[1]   | char*
     *                 |  ...              | char*
     *                 |  NULL             | char*
     * out_auxv +--->  |  <space for auxv> |
     *                 |  envp[0] string   |
     *                 |  envp[1] string   |
     *                 |  ...              |
     *                 |  argv[0] string   |
     *                 |  argv[1] string   |
     *                 |  ...              |
     *                 +-------------------+
     */
    size_t argc      = 0;
    size_t argv_size = 0;
    for (const char** a = argv; *a; a++) {
        argv_size += strlen(*a) + 1;
        argc++;
    }

    /* we populate the stack memory region from two ends:
     *   - memory at high addresses contains buffers with argv + envp strings
     *   - memory at low addresses contains argc and pointer-arrays of argv, envp, and auxv */
    long* argc_ptr = ALLOCATE_FROM_LOW_ADDR(sizeof(long));
    *argc_ptr = argc;

    /* pre-allocate enough space to hold all argv strings */
    char* argv_str = ALLOCATE_FROM_HIGH_ADDR(argv_size);

    /* Even though the SysV ABI does not specify the order of argv strings, some applications
     * (notably Node.js's libuv) assume the compact encoding of argv where (1) all strings are
     * located adjacently and (2) in increasing order. */
    const char** new_argv = stack_low_addr;
    for (const char** a = argv; *a; a++) {
        size_t size = strlen(*a) + 1;
        const char** argv_ptr = ALLOCATE_FROM_LOW_ADDR(sizeof(const char*));  /* ptr to argv[i] */
        memcpy(argv_str, *a, size);                                           /* argv[i] string */
        *argv_ptr = argv_str;
        argv_str += size;
    }
    *((const char**)ALLOCATE_FROM_LOW_ADDR(sizeof(const char*))) = NULL;

    /* populate envp on stack similarly to argv */
    size_t envp_size = 0;
    for (const char** e = envp; *e; e++) {
        envp_size += strlen(*e) + 1;
    }
    char* envp_str = ALLOCATE_FROM_HIGH_ADDR(envp_size);

    const char** new_envp = stack_low_addr;
    for (const char** e = envp; *e; e++) {
        size_t size = strlen(*e) + 1;
        const char** envp_ptr = ALLOCATE_FROM_LOW_ADDR(sizeof(const char*)); /* ptr to envp[i] */
        memcpy(envp_str, *e, size);                                          /* envp[i] string */
        *envp_ptr = envp_str;
        envp_str += size;
    }
    *((const char**)ALLOCATE_FROM_LOW_ADDR(sizeof(const char*))) = NULL;

    /* reserve space for ELF aux vectors, populated later in execute_elf_object() */
    elf_auxv_t* new_auxv = ALLOCATE_FROM_LOW_ADDR(REQUIRED_ELF_AUXV * sizeof(elf_auxv_t) +
                                                  REQUIRED_ELF_AUXV_SPACE);

    /* we have now low part of stack (with argc and pointer-arrays of argv, envp, auxv), high part
     * of stack (with argv and envp strings) and an empty space in the middle: we must remove the
     * empty middle by moving the low part of stack adjacent to the high part */
    size_t move_size         = stack_low_addr - stack;
    void* new_stack_low_addr = stack_high_addr - move_size;

    /* x86-64 SysV ABI requires 16B alignment of stack on ELF entrypoint */
    new_stack_low_addr = ALIGN_DOWN_PTR(new_stack_low_addr, 16UL);
    memmove(new_stack_low_addr, stack, move_size);

    /* pointer-arrays of argv, envp, and auxv were allocated on low part of stack and shifted via
     * memmove above, need to shift pointers to their bases */
    size_t shift = new_stack_low_addr - stack;
    new_argv = (void*)new_argv + shift;
    new_envp = (void*)new_envp + shift;
    new_auxv = (void*)new_auxv + shift;

    /* clear working area at the bottom */
    memset(stack, 0, shift);

    /* TODO: remove this, but see the comment in `shim_do_execve`. */
    /* set global envp pointer for future checkpoint/migration: this is required for fork/clone
     * case (so that migrated envp points to envvars on the migrated stack) and redundant for
     * execve case (because execve passes an explicit list of envvars to child process) */
    migrated_envp = new_envp;

    *out_argp = new_stack_low_addr;
    *out_auxv = new_auxv;
    return 0;
}

int init_stack(const char** argv, const char** envp, const char*** out_argp,
               elf_auxv_t** out_auxv) {
    int ret;

    assert(g_manifest_root);
    uint64_t stack_size;
    ret = toml_sizestring_in(g_manifest_root, "sys.stack.size", get_rlimit_cur(RLIMIT_STACK),
                             &stack_size);
    if (ret < 0) {
        log_error("Cannot parse 'sys.stack.size'");
        return -EINVAL;
    }

    stack_size = ALLOC_ALIGN_UP(stack_size);
    set_rlimit_cur(RLIMIT_STACK, stack_size);

    struct shim_thread* cur_thread = get_cur_thread();
    if (!cur_thread || cur_thread->stack)
        return 0;

    void* stack = allocate_stack(stack_size, ALLOC_ALIGNMENT, /*user=*/true);
    if (!stack)
        return -ENOMEM;

    /* if there is envp inherited from parent, use it */
    envp = migrated_envp ?: envp;

    ret = populate_stack(stack, stack_size, argv, envp, out_argp, out_auxv);
    if (ret < 0)
        return ret;

    cur_thread->stack_top = stack + stack_size;
    cur_thread->stack     = stack;
    cur_thread->stack_red = stack - ALLOC_ALIGNMENT;
    return 0;
}

static int read_environs(const char** envp) {
    for (const char** e = envp; *e; e++) {
        if (strstartswith(*e, "LD_LIBRARY_PATH=")) {
            /* populate `g_library_paths` with entries from LD_LIBRARY_PATH envvar */
            const char* s = *e + static_strlen("LD_LIBRARY_PATH=");
            size_t npaths = 2; // One for the first entry, one for the last NULL.
            for (const char* tmp = s; *tmp; tmp++)
                if (*tmp == ':')
                    npaths++;
            char** paths = malloc(sizeof(const char*) * npaths);
            if (!paths)
                return -ENOMEM;

            size_t cnt = 0;
            while (*s) {
                const char* next;
                for (next = s; *next && *next != ':'; next++)
                    ;
                char* str = alloc_substr(s, next - s);
                if (!str) {
                    for (size_t i = 0; i < cnt; i++)
                        free(paths[i]);
                    free(paths);
                    return -ENOMEM;
                }
                paths[cnt++] = str;
                s = *next ? next + 1 : next;
            }

            paths[cnt] = NULL;

            assert(!g_library_paths);
            g_library_paths = paths;
            return 0;
        }
    }

    return 0;
}

#define CALL_INIT(func, args...) func(args)

#define RUN_INIT(func, ...)                                                  \
    do {                                                                     \
        int _err = CALL_INIT(func, ##__VA_ARGS__);                           \
        if (_err < 0) {                                                      \
            log_error("Error during shim_init() in " #func " (%d)", _err);   \
            DkProcessExit(-_err);                                            \
        }                                                                    \
    } while (0)

noreturn void* shim_init(int argc, const char** argv, const char** envp) {
    g_pal_public_state = DkGetPalPublicState();
    assert(g_pal_public_state);

    g_log_level = g_pal_public_state->log_level;

    /* create the initial TCB, shim can not be run without a tcb */
    shim_tcb_init();

    log_setprefix(shim_get_tcb());

    call_init_array();

    extern const char g_gramine_commit_hash[];
    log_debug("Gramine was built from commit: %s", g_gramine_commit_hash);

    log_debug("Host: %s", g_pal_public_state->host_type);

    if (!IS_POWER_OF_2(ALLOC_ALIGNMENT)) {
        log_error("Error during shim_init(): PAL allocation alignment not a power of 2");
        DkProcessExit(EINVAL);
    }

    g_manifest_root = g_pal_public_state->manifest_root;

    shim_xstate_init();

    if (!create_lock(&__master_lock)) {
        log_error("Error during shim_init(): failed to allocate __master_lock");
        DkProcessExit(ENOMEM);
    }

    RUN_INIT(init_vma);
    RUN_INIT(init_slab);
    RUN_INIT(read_environs, envp);
    RUN_INIT(init_rlimit);
    RUN_INIT(init_fs);
    RUN_INIT(init_fs_lock);
    RUN_INIT(init_dcache);
    RUN_INIT(init_handle);
    RUN_INIT(init_r_debug);

    log_debug("Shim loaded at %p, ready to initialize", &__load_address);

    if (g_pal_public_state->parent_process) {
        struct checkpoint_hdr hdr;

        int ret = read_exact(g_pal_public_state->parent_process, &hdr, sizeof(hdr));
        if (ret < 0) {
            log_error("shim_init: failed to read the whole checkpoint header: %d", ret);
            DkProcessExit(1);
        }

        assert(hdr.size);
        RUN_INIT(receive_checkpoint_and_restore, &hdr);
    } else {
        g_process_ipc_ids.self_vmid = STARTING_VMID;
    }

    RUN_INIT(init_mount_root);
    RUN_INIT(init_ipc);
    RUN_INIT(init_process, argc, argv);
    RUN_INIT(init_threading);
    RUN_INIT(init_mount);
    RUN_INIT(init_important_handles);

    /* Update log prefix after we initialized `g_process.exec` */
    log_setprefix(shim_get_tcb());

    RUN_INIT(init_async_worker);

    const char** new_argp;
    elf_auxv_t* new_auxv;
    RUN_INIT(init_stack, argv, envp, &new_argp, &new_auxv);

    /* TODO: Support running non-ELF executables (scripts) */
    RUN_INIT(init_elf_objects);
    RUN_INIT(init_signal_handling);
    RUN_INIT(init_ipc_worker);

    if (g_pal_public_state->parent_process) {
        int ret = connect_to_process(g_process_ipc_ids.parent_vmid);
        if (ret < 0) {
            log_error("shim_init: failed to establish IPC connection to parent: %d", ret);
            DkProcessExit(1);
        }

        /* Send a dummy request causing the IPC leader to connect to this process, so that it is
         * included in all broadcast messages. */
        IDTYPE dummy = 0;
        ret = ipc_get_id_owner(/*id=*/0, /*out_owner=*/&dummy);
        if (ret < 0) {
            log_debug("shim_init: failed to get a connection from IPC leader to us: %d", ret);
            DkProcessExit(1);
        }
        assert(dummy == 0); // Nobody should own ID `0`.

        /* Notify the parent process we are done. */
        char dummy_c = 0;
        ret = write_exact(g_pal_public_state->parent_process, &dummy_c, sizeof(dummy_c));
        if (ret < 0) {
            log_error("shim_init: failed to write ready notification: %d", ret);
            DkProcessExit(1);
        }

        /* Wait for parent to settle its adult things. */
        ret = read_exact(g_pal_public_state->parent_process, &dummy_c, sizeof(dummy_c));
        if (ret < 0) {
            log_error("shim_init: failed to read parent's confirmation: %d", ret);
            DkProcessExit(1);
        }
    } else { /* !g_pal_public_state->parent_process */
        RUN_INIT(init_sync_server);
    }

    /* Note that in the main process, we initialize both sync server and sync client, and the client
     * communicates with server over a "loopback" IPC connection. */
    RUN_INIT(init_sync_client);

    log_debug("Shim process initialized");

    shim_tcb_t* cur_tcb = shim_get_tcb();

    if (cur_tcb->context.regs) {
        restore_child_context_after_clone(&cur_tcb->context);
        /* UNREACHABLE */
    }

    set_default_tls();

    /* At this point, the exec map has been either copied from checkpoint, or initialized in
     * `init_loader`. */
    execute_elf_object(/*exec_map=*/NULL, new_argp, new_auxv);
    /* UNREACHABLE */
}

static int get_256b_random_hex_string(char* buf, size_t size) {
    char random[32]; /* 256-bit random value, sufficiently crypto secure */

    if (size < sizeof(random) * 2 + 1)
        return -ENOMEM;

    int ret = DkRandomBitsRead(&random, sizeof(random));
    if (ret < 0)
        return pal_to_unix_errno(ret);

    BYTES2HEXSTR(random, buf, size);
    return 0;
}

int vmid_to_uri(IDTYPE vmid, char* uri, size_t uri_len) {
    int ret = snprintf(uri, uri_len, URI_PREFIX_PIPE "%u", vmid);
    if (ret < 0 || (size_t)ret >= uri_len) {
        return -ERANGE;
    }
    return 0;
}

int create_pipe(char* name, char* uri, size_t size, PAL_HANDLE* hdl, bool use_vmid_for_name) {
    int ret;
    size_t len;
    char pipename[PIPE_URI_SIZE];
    PAL_HANDLE pipe = NULL;

    assert(hdl);
    assert(uri);
    assert(size);

    while (true) {
        if (use_vmid_for_name) {
            len = snprintf(pipename, sizeof(pipename), "%u", g_process_ipc_ids.self_vmid);
            if (len >= sizeof(pipename))
                return -ERANGE;
        } else {
            ret = get_256b_random_hex_string(pipename, sizeof(pipename));
            if (ret < 0)
                return ret;
        }

        log_debug("Creating pipe: " URI_PREFIX_PIPE_SRV "%s", pipename);
        len = snprintf(uri, size, URI_PREFIX_PIPE_SRV "%s", pipename);
        if (len >= size)
            return -ERANGE;

        ret = DkStreamOpen(uri, PAL_ACCESS_RDWR, /*share_flags=*/0, PAL_CREATE_IGNORED,
                           /*options=*/0, &pipe);
        if (ret < 0) {
            if (!use_vmid_for_name && ret == -PAL_ERROR_STREAMEXIST) {
                /* tried to create a pipe with random name but it already exists */
                continue;
            }
            return pal_to_unix_errno(ret);
        }

        break; /* succeeded in creating the pipe with random/vmid name */
    }

    /* output generated pipe handle, URI, and name */
    *hdl = pipe;
    len = snprintf(uri, size, URI_PREFIX_PIPE "%s", pipename);
    if (name)
        memcpy(name, pipename, sizeof(pipename));
    return 0;
}
