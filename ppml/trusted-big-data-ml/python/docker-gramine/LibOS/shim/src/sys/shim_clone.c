/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include <errno.h>
#include <linux/sched.h>

#include "pal.h"
#include "shim_checkpoint.h"
#include "shim_fs.h"
#include "shim_internal.h"
#include "shim_ipc.h"
#include "shim_lock.h"
#include "shim_table.h"
#include "shim_thread.h"
#include "shim_types.h"
#include "shim_vma.h"

struct shim_clone_args {
    PAL_HANDLE create_event;
    PAL_HANDLE initialize_event;
    struct shim_thread* thread;
    void* stack;
    unsigned long tls;
    PAL_CONTEXT* regs;
};

/*
 * This Function is a wrapper around the user provided function.
 * Code flow for clone is as follows -
 * 1) User application allocates stack for child process and
 *    calls clone. The clone code sets up the user function
 *    address and the argument address on the child stack.
 * 2)we Hijack the clone call and control flows to shim_clone
 * 3)In Shim Clone we just call the DK Api to create a thread by providing a
 *   wrapper function around the user provided function
 * 4)PAL layer allocates a stack and then invokes the clone syscall
 * 5)PAL runs thread_init function on PAL allocated Stack
 * 6)thread_init calls our wrapper and gives the user provided stack
 *   address.
 * 7.In the wrapper function ,we just do the stack switch to user
 *   Provided stack and execute the user Provided function.
 */
static int clone_implementation_wrapper(void* arg_) {
    /* The child thread created by PAL is now running on the PAL allocated stack. We need to switch
     * to the user provided stack. */

    /* We acquired ownership of arg->thread from the caller, hence there is
     * no need to call get_thread. */
    struct shim_clone_args* arg = (struct shim_clone_args*)arg_;
    struct shim_thread* my_thread = arg->thread;
    assert(my_thread);

    shim_tcb_init();
    set_cur_thread(my_thread);

    /* only now we can call LibOS/PAL functions because they require a set-up TCB;
     * do not move the below functions before shim_tcb_init/set_cur_thread()! */
    object_wait_with_retry(arg->create_event);
    DkObjectClose(arg->create_event);

    shim_tcb_t* tcb = my_thread->shim_tcb;

    log_setprefix(tcb);

    if (my_thread->set_child_tid) {
        *(my_thread->set_child_tid) = my_thread->tid;
        my_thread->set_child_tid = NULL;
    }

    void* stack = arg->stack;

    struct shim_vma_info vma_info;
    if (lookup_vma(ALLOC_ALIGN_DOWN_PTR(stack), &vma_info) < 0) {
        return -EFAULT;
    }
    my_thread->stack_top = (char*)vma_info.addr + vma_info.length;
    my_thread->stack_red = my_thread->stack = vma_info.addr;
    if (vma_info.file) {
        put_handle(vma_info.file);
    }

    add_thread(my_thread);

    /* Copy regs before we let the parent release them. */
    PAL_CONTEXT regs;
    pal_context_copy(&regs, arg->regs);
    pal_context_set_sp(&regs, (unsigned long)stack);
    tcb->context.regs = &regs;
    tcb->context.tls = arg->tls;

    /* Inform the parent thread that we finished initialization. */
    DkEventSet(arg->initialize_event);

    put_thread(my_thread);

    restore_child_context_after_clone(&tcb->context);
}

static BEGIN_MIGRATION_DEF(fork, struct shim_process* process_description,
                           struct shim_thread* thread_description,
                           struct shim_ipc_ids* process_ipc_ids) {
    DEFINE_MIGRATE(process_ipc_ids, process_ipc_ids, sizeof(*process_ipc_ids));
    DEFINE_MIGRATE(all_encrypted_files_keys, NULL, 0);
    DEFINE_MIGRATE(dentry_root, NULL, 0);
    DEFINE_MIGRATE(all_mounts, NULL, 0);
    DEFINE_MIGRATE(all_vmas, NULL, 0);
    DEFINE_MIGRATE(process_description, process_description, sizeof(*process_description));
    DEFINE_MIGRATE(thread, thread_description, sizeof(*thread_description));
    DEFINE_MIGRATE(migratable, NULL, 0);
    DEFINE_MIGRATE(brk, NULL, 0);
    DEFINE_MIGRATE(loaded_elf_objects, NULL, 0);
    DEFINE_MIGRATE(topo_info, NULL, 0);
#ifdef DEBUG
    DEFINE_MIGRATE(gdb_map, NULL, 0);
#endif
}
END_MIGRATION_DEF(fork)

static int migrate_fork(struct shim_cp_store* store, struct shim_process* process_description,
                        struct shim_thread* thread_description,
                        struct shim_ipc_ids* process_ipc_ids, va_list ap) {
    __UNUSED(ap);
    /* Take `g_dcache_lock` for the whole checkpointing operation, so that we can access data from
     * dentries. We recursively checkpoint various connected structures, so it's not practical to
     * take the lock just for some part of this operation. */
    lock(&g_dcache_lock);
    int ret = START_MIGRATE(store, fork, process_description, thread_description, process_ipc_ids);
    unlock(&g_dcache_lock);
    return ret;
}

static long do_clone_new_vm(IDTYPE child_vmid, unsigned long flags, struct shim_thread* thread,
                            unsigned long tls, unsigned long user_stack_addr, int* set_parent_tid) {
    assert(!(flags & CLONE_VM));

    struct shim_child_process* child_process = create_child_process();
    if (!child_process) {
        return -ENOMEM;
    }

    struct shim_thread* self = get_cur_thread();

    /* Associate new cpu context to the new process (its main and only thread) for migration
     * since we might need to modify some registers. */
    shim_tcb_t shim_tcb = { 0 };
    __shim_tcb_init(&shim_tcb);
    /* We are copying our own tcb, which should be ok to do, even without any locks. Note this is
     * a shallow copy, so `shim_tcb.context.regs` will be shared with the parent. */
    shim_tcb.context.regs = self->shim_tcb->context.regs;
    shim_tcb.context.tls = tls;

    thread->shim_tcb = &shim_tcb;

    unsigned long parent_stack = 0;
    if (user_stack_addr) {
        struct shim_vma_info vma_info;
        if (lookup_vma((void*)ALLOC_ALIGN_DOWN(user_stack_addr), &vma_info) < 0) {
            destroy_child_process(child_process);
            return -EFAULT;
        }
        thread->stack_top = (char*)vma_info.addr + vma_info.length;
        thread->stack_red = thread->stack = vma_info.addr;
        parent_stack = pal_context_get_sp(self->shim_tcb->context.regs);
        pal_context_set_sp(thread->shim_tcb->context.regs, user_stack_addr);

        if (vma_info.file) {
            put_handle(vma_info.file);
        }
    }

    lock(&g_process.fs_lock);
    struct shim_process process_description = {
        .pid = thread->tid,
        .ppid = g_process.pid,
        .pgid = __atomic_load_n(&g_process.pgid, __ATOMIC_ACQUIRE),
        .root = g_process.root,
        .cwd = g_process.cwd,
        .umask = g_process.umask,
        .exec = g_process.exec,
    };

    get_dentry(process_description.root);
    get_dentry(process_description.cwd);
    get_handle(process_description.exec);

    unlock(&g_process.fs_lock);

    INIT_LISTP(&process_description.children);
    INIT_LISTP(&process_description.zombies);

    clear_lock(&process_description.fs_lock);
    clear_lock(&process_description.children_lock);

    child_process->pid = process_description.pid;
    child_process->child_termination_signal = flags & CSIGNAL;
    child_process->uid = thread->uid;
    child_process->vmid = child_vmid;

    long ret = create_process_and_send_checkpoint(&migrate_fork, child_process,
                                                  &process_description, thread);

    if (parent_stack) {
        pal_context_set_sp(self->shim_tcb->context.regs, parent_stack);
    }

    thread->shim_tcb = NULL;

    put_handle(process_description.exec);
    put_dentry(process_description.cwd);
    put_dentry(process_description.root);

    if (ret >= 0) {
        if (set_parent_tid) {
            *set_parent_tid = thread->tid;
        }

        ret = process_description.pid;
    } else {
        /* Child creation failed, so it was not added to the children list. */
        destroy_child_process(child_process);
    }

    return ret;
}

long shim_do_clone(unsigned long flags, unsigned long user_stack_addr, int* parent_tidptr,
                  int* child_tidptr, unsigned long tls) {
    /*
     * Currently not supported:
     * CLONE_PARENT
     * CLONE_IO
     * CLONE_PIDFD
     * CLONE_NEWNS and friends
     */
    const unsigned long supported_flags =
        CLONE_CHILD_CLEARTID |
        CLONE_CHILD_SETTID |
        CLONE_DETACHED |
        CLONE_FILES |
        CLONE_FS |
        CLONE_PARENT_SETTID |
        CLONE_PTRACE |
        CLONE_SETTLS |
        CLONE_SIGHAND |
        CLONE_SYSVSEM |
        CLONE_THREAD |
        CLONE_UNTRACED |
        CLONE_VFORK |
        CLONE_VM |
        CSIGNAL;

    if (flags & ~supported_flags) {
        log_warning("clone called with unsupported flags argument.");
        return -EINVAL;
    }

    /* CLONE_DETACHED is deprecated and ignored. */
    flags &= ~CLONE_DETACHED;

    /* These 2 flags modify ptrace behavior and can be ignored in Gramine. */
    flags &= ~(CLONE_PTRACE | CLONE_UNTRACED);

    if ((flags & CLONE_THREAD) && !(flags & CLONE_SIGHAND))
        return -EINVAL;
    if ((flags & CLONE_SIGHAND) && !(flags & CLONE_VM))
        return -EINVAL;

    /* Explicitly disallow CLONE_VM without either of CLONE_THREAD or CLONE_VFORK in Gramine. While
     * the Linux allows for such combinations, they do not happen in the wild, so they are
     * explicitly disallowed for now. */
    if (flags & CLONE_VM) {
        if (!((flags & CLONE_THREAD) || (flags & CLONE_VFORK))) {
            log_warning("CLONE_VM without either CLONE_THREAD or CLONE_VFORK is unsupported");
            return -EINVAL;
        }
    }

    if (flags & CLONE_VFORK) {
        /* Instead of trying to support Linux semantics for vfork() -- which requires adding
         * corner-cases in signal handling and syscalls -- we simply treat vfork() as fork(). We
         * assume that performance hit is negligible (Gramine has to migrate internal state anyway
         * which is slow) and apps do not rely on insane Linux-specific semantics of vfork().  */
        log_warning("vfork was called by the application, implemented as an alias to fork in "
                    "Gramine");
        flags &= ~(CLONE_VFORK | CLONE_VM);
    }

    if (!(flags & CLONE_VM)) {
        /* If thread/process does not share VM we cannot handle these flags. */
        if (flags & (CLONE_FILES | CLONE_FS | CLONE_SYSVSEM)) {
            return -EINVAL;
        }
    } else {
        /* If it does share VM, we currently assume these flags are set. Supposedly erroring out
         * here would break too many applications ... */
        // TODO: either implement these flags (shouldn't be hard) or return an error
        flags |= CLONE_FS | CLONE_SYSVSEM;
    }

    int* set_parent_tid = NULL;
    if (flags & CLONE_PARENT_SETTID) {
        if (!parent_tidptr)
            return -EINVAL;
        set_parent_tid = parent_tidptr;
    }

    bool clone_new_process = false;
    if (!(flags & CLONE_VM)) {
        /* New process has its own address space - currently in Gramine that means it's just
         * another process. */
        assert(!(flags & CLONE_THREAD));
        clone_new_process = true;
    }

    long ret = 0;

    struct shim_thread* thread = get_new_thread();
    if (!thread) {
        ret = -ENOMEM;
        goto failed;
    }

    if (flags & CLONE_CHILD_SETTID) {
        if (!child_tidptr) {
            ret = -EINVAL;
            goto failed;
        }
        thread->set_child_tid = child_tidptr;
    }

    if (flags & CLONE_CHILD_CLEARTID)
        thread->clear_child_tid = child_tidptr;

    if (!(flags & CLONE_SETTLS)) {
        tls = get_tls();
    }

    IDTYPE new_vmid = 0;
    if (clone_new_process) {
        ret = ipc_get_new_vmid(&new_vmid);
        if (ret < 0) {
            log_error("Cound not allocate new vmid!");
            ret = -EAGAIN;
            goto failed;
        }
    }

    IDTYPE tid = get_new_id(/*move_ownership_to=*/new_vmid);
    if (!tid) {
        log_error("Could not allocate a tid!");
        ret = -EAGAIN;
        goto failed;
    }
    thread->tid = tid;

    if (clone_new_process) {
        ret = do_clone_new_vm(new_vmid, flags, thread, tls, user_stack_addr, set_parent_tid);

        /* We should not have saved any references to this thread anywhere and `put_thread` below
         * should free it. */
        assert(__atomic_load_n(&thread->ref_count.counter, __ATOMIC_RELAXED) == 1);

        /* We do not own `thread->tid` aka `tid` anymore, clean it so that `put_thread` does not
         * try to release it. */
        thread->tid = 0;
        put_thread(thread);

        if (ret < 0) {
            /* The child process have already taken the ownership of `tid`, let's change it back. */
            int tmp_ret = ipc_change_id_owner(tid, g_process_ipc_ids.self_vmid);
            if (tmp_ret < 0) {
                log_debug("Failed to change back ID %u owner: %d", tid, tmp_ret);
                /* No way to recover gracefully. */
                DkProcessExit(1);
            }
            /* ... and release it. */
            tmp_ret = ipc_release_id_range(tid, tid);
            if (tmp_ret < 0) {
                log_debug("Failed to release ID %u: %d", tid, tmp_ret);
                /* No way to recover gracefully. */
                DkProcessExit(1);
            }
        }
        return ret;
    }

    assert(flags & CLONE_THREAD);

    ret = alloc_thread_libos_stack(thread);
    if (ret < 0) {
        goto failed;
    }

    /* Threads do not generate signals on death, ignore it. */
    flags &= ~CSIGNAL;

    if (!(flags & CLONE_FILES)) {
        /* If CLONE_FILES is not given, the new thread should receive its own copy of the
         * descriptors table. */
        struct shim_handle_map* new_map = NULL;

        dup_handle_map(&new_map, thread->handle_map);
        set_handle_map(thread, new_map);
        put_handle_map(new_map);
    }

    /* Currently CLONE_SIGHAND is always set here, since CLONE_VM implies CLONE_THREAD (which
     * implies CLONE_SIGHAND). */
    assert(flags & CLONE_SIGHAND);

    enable_locking();

    struct shim_clone_args new_args;
    memset(&new_args, 0, sizeof(new_args));

    ret = DkEventCreate(&new_args.create_event, /*init_signaled=*/false, /*auto_clear=*/false);
    if (ret < 0) {
        ret = pal_to_unix_errno(ret);
        goto clone_thread_failed;
    }

    ret = DkEventCreate(&new_args.initialize_event, /*init_signaled=*/false, /*auto_clear=*/false);
    if (ret < 0) {
        ret = pal_to_unix_errno(ret);
        goto clone_thread_failed;
    }

    struct shim_thread* self = get_cur_thread();

    /* Increasing refcount due to copy below. Passing ownership of the new copy
     * of this pointer to the new thread (receiver of new_args). */
    get_thread(thread);
    new_args.thread = thread;
    new_args.stack  = (void*)(user_stack_addr ?: pal_context_get_sp(self->shim_tcb->context.regs));
    new_args.tls    = tls;
    new_args.regs   = self->shim_tcb->context.regs;

    // Invoke DkThreadCreate to spawn off a child process using the actual
    // "clone" system call. DkThreadCreate allocates a stack for the child
    // and then runs the given function on that stack However, we want our
    // child to run on the Parent allocated stack , so once the DkThreadCreate
    // returns .The parent comes back here - however, the child is Happily
    // running the function we gave to DkThreadCreate.
    PAL_HANDLE pal_handle = NULL;
    ret = DkThreadCreate(clone_implementation_wrapper, &new_args, &pal_handle);
    if (ret < 0) {
        ret = pal_to_unix_errno(ret);
        put_thread(new_args.thread);
        goto clone_thread_failed;
    }

    thread->pal_handle = pal_handle;

    if (set_parent_tid)
        *set_parent_tid = thread->tid;

    DkEventSet(new_args.create_event);
    object_wait_with_retry(new_args.initialize_event);
    DkObjectClose(new_args.initialize_event); // TODO: handle errors

    put_thread(thread);
    return tid;

clone_thread_failed:
    if (new_args.create_event)
        DkObjectClose(new_args.create_event);
    if (new_args.initialize_event)
        DkObjectClose(new_args.initialize_event);
failed:
    if (thread)
        put_thread(thread);
    return ret;
}
