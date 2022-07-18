/*
 * RPC threads are helper threads that run in untrusted mode alongside enclave threads. RPC threads
 * issue system calls on behalf of enclave threads. This allows "exitless" design when app threads
 * never leave the enclave (except for a few syscalls where there is no benefit, e.g., nanosleep).
 *
 * "Exitless" design alleviates expensive OCALLs/ECALLs. This was first proposed by SCONE (by
 * Arnautov et al at OSDI 2016) and by Eleos (by Orenbach et al at EuroSys 2017).
 *
 * Brief description: user must specify "sgx.rpc_thread_num = 2" in manifest to create two RPC
 * threads. If user specifies "0" or omits this directive, then no RPC threads are created and all
 * syscalls perform an enclave exit (as in previous versions of Gramine).
 *
 * All enclave and RPC threads work on a single shared RPC queue (global variable `g_rpc_queue`).
 * To issue a syscall, enclave thread enqueues syscall request in the queue and spins waiting for
 * result. RPC threads spin waiting for syscall requests; when request comes, first lucky RPC
 * thread grabs request, issues syscall to OS, and notifies enclave thread by releasing the request
 * lock. RPC queue is implemented as a FIFO ring buffer with one global lock.
 *
 * The RPC queue with its ring buffer resides in *untrusted memory*. The enclave code accessing the
 * RPC queue must be carefully written to withstand attacks tampering with the queue.
 *
 * RPC queue can have up to RPC_QUEUE_SIZE requests simultaneously. All requests are allocated on
 * the untrusted stack of the enclave thread; enclave thread owns its requests and pops them off
 * stack when done with the system call. After enqueuing the request, enclave thread first spins
 * for some time in hope the system call returns immediately (fast path), then sleeps waiting on
 * futex (slow path, useful for blocking syscalls).
 *
 * NOTE: number of created RPC threads must match max number of simultaneous enclave threads. If
 * there are more RPC threads, CPU time is wasted. If there are less, some enclave threads may
 * starve, especially if there are many blocking syscalls by other enclave threads.
 *
 * NOTE: The Exitless feature trades slow OCALLs/ECALLs for fast RPC-queue communication at the
 * cost of occupying more CPU cores and burning more CPU cycles. For example, a single-threaded
 * Redis instance on Linux becomes 5-threaded on Gramine-SGX. Therefore, Exitless may negatively
 * impact throughput but may improve latency. Only a subset of applications may benefit from
 * Exitless, in particular, single-threaded latency-sensitive apps that cannot be parallelized.
 *
 * Prototype code was written by Meni Orenbach and adapted to Gramine by Dmitrii Kuvaiskii.
 */
#ifndef QUEUE_H_
#define QUEUE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "spinlock.h"

/* Number of iterations to spin before sleeping. We choose 1M as follows: we want to sleep on
 * blocking syscalls but we want to allow ample time for fast syscalls to complete. We choose
 * 1 millisecond -- more than enough time to complete any non-blocking syscall. Assuming a 1GHz
 * CPU and no pipelining (and ignoring the pause instruction), 1 millisecond is 1M cycles. This
 * works well in practice. */
#define RPC_SPINLOCK_TIMEOUT 1000000

#define RPC_QUEUE_SIZE  1024 /* max # of requests in RPC queue */
#define MAX_RPC_THREADS 256  /* max number of RPC threads */

typedef struct {
    spinlock_t lock; /* can be UNLOCKED / LOCKED_NO_WAITERS / LOCKED_WITH_WAITERS */
    long result;
    uint64_t ocall_index;
    void* buffer;
} rpc_request_t;

typedef struct rpc_queue {
    spinlock_t lock;                  /* global lock for enclave and RPC threads */
    uint64_t front, rear;             /* indexes into front and rear ends of q */
    rpc_request_t* q[RPC_QUEUE_SIZE]; /* queue of syscall requests */
    int rpc_threads[MAX_RPC_THREADS]; /* RPC threads (thread IDs) */
    size_t rpc_threads_cnt;           /* number of RPC threads */
} rpc_queue_t;

extern rpc_queue_t* g_rpc_queue;  /* global RPC queue */

static inline void rpc_queue_init(rpc_queue_t* q) {
    spinlock_init(&q->lock);
    q->front = 0;
    q->rear  = 0;
    for (size_t i = 0; i < RPC_QUEUE_SIZE; i++)
        q->q[i] = NULL;
}

/*!
 * \brief Enqueue OCALL request `req` in the shared RPC queue `q`.
 *
 * This function is called from the enclave code and thus must be written carefully to withstand
 * attacks tampering with untrusted `req` and untrusted `q`. In particular, `req` and `q` must not
 * have arbitrary pointers (or alternatively the code below must sanitize possible pointer values)
 * to prevent arbitrary writes to/reads from the enclave memory. Similarly, `q->q[idx]` code must
 * ensure that `idx` points inside the `q->q` array to prevent buffer overflows.
 */
static inline bool rpc_enqueue(rpc_queue_t* q, rpc_request_t* req) {
    bool ret = false;
    spinlock_lock(&q->lock);

    if (q->rear - q->front >= RPC_QUEUE_SIZE) {
        /* queue is full, cannot enqueue */
        goto out;
    }

    if (q->q[q->rear % RPC_QUEUE_SIZE]) {
        /* current slot is occupied, cannot enqueue but the caller can try again */
        q->rear++;
        goto out;
    }

    q->q[q->rear % RPC_QUEUE_SIZE] = req;
    q->rear++;
    ret = true;
out:
    spinlock_unlock(&q->lock);
    return ret;
}

/*!
 * \brief Dequeue OCALL request `req` from the shared RPC queue `q`.
 *
 * This function is called only from the untrusted code and thus has no security implications.
 */
static inline rpc_request_t* rpc_dequeue(rpc_queue_t* q) {
    if (__atomic_load_n(&q->front, __ATOMIC_RELAXED) ==
            __atomic_load_n(&q->rear,  __ATOMIC_RELAXED)) {
        /* quick check that queue is empty; this doesn't acquire a spinlock and thus lowers latency
         * on rpc_enqueue() performed by enclave threads (they don't have to wait for spinlock
         * release); note that untrusted RPC threads will simply retry again if this check fails */
        return NULL;
    }

    rpc_request_t* ret = NULL;
    spinlock_lock(&q->lock);

    if (q->front == q->rear) {
        /* queue is empty, nothing to dequeue */
        goto out;
    }

    if (!q->q[q->front % RPC_QUEUE_SIZE]) {
        /* current slot is empty, cannot dequeue but the caller can try next one */
        q->front++;
        goto out;
    }

    ret = q->q[q->front % RPC_QUEUE_SIZE];
    q->q[q->front % RPC_QUEUE_SIZE] = NULL;
    q->front++;
out:
    spinlock_unlock(&q->lock);
    return ret;
}

#endif /* QUEUE_H_ */
