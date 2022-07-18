#include <stdatomic.h>
#include <stdnoreturn.h>

#include "pal.h"
#include "pal_regression.h"

const char* private1 = "Hello World 1";
const char* private2 = "Hello World 2";

static atomic_int count = 0;

static noreturn int callback(void* args) {
    pal_printf("Run in Child Thread: %s\n", (char*)args);

    while (count < 10) {
        while (!(count % 2)) {
            DkThreadYieldExecution();
        }
        count++;
    }

    pal_printf("Threads Run in Parallel OK\n");

    if (DkSegmentBaseSet(PAL_SEGMENT_FS, (uintptr_t)&private2) < 0) {
        pal_printf("Failed to set FS\n");
        DkThreadExit(/*clear_child_tid=*/NULL);
    }

    const char* ptr2;
    __asm__ volatile("mov %%fs:0, %0" : "=r"(ptr2)::"memory");
    pal_printf("Private Message (FS Segment) 2: %s\n", ptr2);

    count = 100;
    DkThreadExit(/*clear_child_tid=*/NULL);
    /* UNREACHABLE */
}

int main(void) {
    if (DkSegmentBaseSet(PAL_SEGMENT_FS, (uintptr_t)&private1) < 0) {
        pal_printf("Failed to set FS\n");
        return 1;
    }
    const char* ptr1;
    __asm__ volatile("mov %%fs:0, %0" : "=r"(ptr1)::"memory");
    pal_printf("Private Message (FS Segment) 1: %s\n", ptr1);

    PAL_HANDLE thread1 = NULL;
    char arg[] = "Hello World";
    int ret = DkThreadCreate(callback, arg, &thread1);
    if (ret < 0)
        return 1;

    pal_printf("Child Thread Created\n");

    while (count < 9) {
        while (!!(count % 2)) {
            DkThreadYieldExecution();
        }
        count++;
    }

    while (count != 100) {
        DkThreadYieldExecution();
    }
    pal_printf("Child Thread Exited\n");
    return 0;
}
