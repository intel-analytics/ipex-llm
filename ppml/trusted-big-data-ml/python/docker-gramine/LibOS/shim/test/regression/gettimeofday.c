#include <err.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define MAX_TIME_DIFF_SEC 20
#define THREAD_NUM 4
#define ITERATIONS 1000000

struct timeval base_tv;

static void* foo(void* arg) {
    int ret;

    for (size_t i = 0; i < ITERATIONS; i++) {
        struct timeval tv;
        ret = gettimeofday(&tv, NULL);
        if (ret < 0)
            err(1, "thread gettimeofday");

        if (tv.tv_sec < base_tv.tv_sec || tv.tv_sec - base_tv.tv_sec > MAX_TIME_DIFF_SEC)
            errx(1, "Retrieved time is more than 20 seconds away from base time");
    }

    return NULL;
}

int main(int argc, char** argv) {
    int ret;

    ret = gettimeofday(&base_tv, NULL);
    if (ret < 0)
        err(1, "base gettimeofday");

    printf("Starting time: %lu sec, %lu usec\n", base_tv.tv_sec, base_tv.tv_usec);

    pthread_t thread[THREAD_NUM];
    for (size_t i = 0; i < THREAD_NUM; i++) {
        pthread_create(&thread[i], NULL, foo, NULL);
    }
    for (size_t i = 0; i < THREAD_NUM; i++) {
        pthread_join(thread[i], NULL);
    }

    struct timeval end_tv;
    ret = gettimeofday(&end_tv, NULL);
    if (ret < 0)
        err(1, "base gettimeofday");

    uint64_t sec_diff  = end_tv.tv_sec - base_tv.tv_sec -
                         (end_tv.tv_usec < base_tv.tv_usec ? 1 : 0);
    uint64_t usec_diff = end_tv.tv_usec > base_tv.tv_usec ? end_tv.tv_usec - base_tv.tv_usec
                                                          : base_tv.tv_usec - end_tv.tv_usec;
    printf("Finish time: %lu sec, %lu usec (passed: %lu sec, %lu usec)\n", end_tv.tv_sec,
           end_tv.tv_usec, sec_diff, usec_diff);
    puts("TEST OK");
    return 0;
}
