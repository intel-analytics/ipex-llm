#include <stdatomic.h>

#include "pal.h"
#include "pal_error.h"
#include "pal_regression.h"

volatile bool dummy_true = true;

static atomic_bool thread2_started = false;
static atomic_bool thread3_started = false;
static atomic_bool thread3_exit_ok = true;
static atomic_bool thread4_started = false;

static int thread2_run(void* args) {
    pal_printf("Thread 2 started.\n");

    thread2_started = true;

    pal_printf("Exiting thread 2 by return.\n");
    return 0;
}

static int thread3_run(void* args) {
    pal_printf("Thread 3 started.\n");

    thread3_started = true;

    pal_printf("Exiting thread 3 by DkThreadExit.\n");

    // Ensure that the compiler can't know that this should never return.
    if (dummy_true) {
        DkThreadExit(/*clear_child_tid=*/NULL);
        /* UNREACHABLE */
    }

    thread3_exit_ok = false;
    pal_printf("Exiting thread 3 failed.\n");

    return 0;
}

static int thread4_run(void* args) {
    pal_printf("Thread 4 started.\n");

    thread4_started = true;

    pal_printf("Exiting thread 4 by return.\n");
    return 0;
}

// If there's a thread limit, like on SGX, it should be set to exactly 2. There
// should be only the main thread and only one other thread at a time.
int main(void) {
    pal_printf("Thread 1 (main) started.\n");

    PAL_HANDLE sleep_handle = NULL;
    if (DkEventCreate(&sleep_handle, /*init_signaled=*/false, /*auto_clear=*/false) < 0) {
        pal_printf("DkEventCreate failed\n");
        return 1;
    }

    PAL_HANDLE thread2 = NULL;
    int ret = DkThreadCreate(thread2_run, NULL, &thread2);
    if (ret < 0) {
        pal_printf("DkThreadCreate failed for thread 2.\n");
        return 1;
    }

    // 1 s should be enough even on a very busy system to start a thread and
    // then exit it again including all cleanup.
    uint64_t timeout = 1000000;
    ret = DkEventWait(sleep_handle, &timeout);
    if (ret != -PAL_ERROR_TRYAGAIN) {
        pal_printf("DkEventWait failed\n");
        return 1;
    }

    if (thread2_started) {
        pal_printf("Thread 2 ok.\n");
    }

    PAL_HANDLE thread3 = NULL;
    ret = DkThreadCreate(thread3_run, NULL, &thread3);
    if (ret < 0) {
        pal_printf("DkThreadCreate failed for thread 3.\n");
        return 1;
    }

    timeout = 1000000;
    ret = DkEventWait(sleep_handle, &timeout);
    if (ret != -PAL_ERROR_TRYAGAIN) {
        pal_printf("DkEventWait failed\n");
        return 1;
    }

    if (thread3_started && thread3_exit_ok) {
        pal_printf("Thread 3 ok.\n");
    }

    PAL_HANDLE thread4 = NULL;
    ret = DkThreadCreate(thread4_run, NULL, &thread4);
    if (ret < 0) {
        pal_printf("DkThreadCreate failed for thread 4.\n");
        return 1;
    }

    timeout = 1000000;
    ret = DkEventWait(sleep_handle, &timeout);
    if (ret != -PAL_ERROR_TRYAGAIN) {
        pal_printf("DkEventWait failed\n");
        return 1;
    }

    if (thread4_started) {
        pal_printf("Thread 4 ok.\n");
    }

    return 0;
}
