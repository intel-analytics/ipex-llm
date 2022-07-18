#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

#include "futex.h"

#define SLEEP_SEC 1

int main(int argc, const char** argv) {
    int myfutex = 0;
    int ret;
    int futex_errno = 0;

    struct timespec t = {.tv_sec = SLEEP_SEC, .tv_nsec = 0};
    struct timeval tv1 = {0};
    struct timeval tv2 = {0};

    printf("invoke futex syscall with a %d second timeout\n", SLEEP_SEC);
    if (gettimeofday(&tv1, NULL)) {
        printf("Cannot get time 1: %m\n");
        return 1;
    }
    ret = syscall(SYS_futex, &myfutex, FUTEX_WAIT, 0, &t, NULL, 0);
    futex_errno = errno;
    if (gettimeofday(&tv2, NULL)) {
        printf("Cannot get time 2: %m\n");
        return 1;
    }

    if (ret != -1 || futex_errno != ETIMEDOUT) {
        printf("futex syscall returned: %d with errno: %d (%s)\n", ret, futex_errno,
               strerror(futex_errno));
        return 1;
    }

    long long diff = (tv2.tv_sec - tv1.tv_sec) * 1000000ll;
    diff += tv2.tv_usec - tv1.tv_usec;

    if (diff < 0) {
        printf("Just moved back in time (%lld), better call Ghostbusters!\n", diff);
        return 1;
    }
    if (diff < 1000000ll * SLEEP_SEC) {
        printf("Slept for %lld microseconds, which is less than %d seconds\n", diff, SLEEP_SEC);
        return 1;
    }

    puts("futex correctly timed out");
    return 0;
}
