#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

static uint64_t rdtsc_start(void) {
    uint32_t low, high;
    __asm__ volatile(
        "mfence; lfence;" /* memory barriers to sync RDTSC */
        "rdtsc;"
        : "=d"(high), "=a"(low)
        :
        : "memory");
    return (low | ((uint64_t)high) << 32);
}

static uint64_t rdtsc_end(void) {
    uint32_t low, high;
    __asm__ volatile(
        "rdtscp;" /* use RDTSCP instead of RDTSC, just for testing */
        "lfence;" /* memory barrier to sync RDTSCP */
        : "=d"(high), "=a"(low)
        :
        : "%rcx", "memory");
    return (low | ((uint64_t)high) << 32);
}

int main(int argc, char** argv) {
    int tries = 0;
    uint64_t start = rdtsc_start();
    while (1) {
        /* sleep may return prematurely, make sure we sleep for 1s at least once */
        unsigned int remaining = sleep(1);
        if (!remaining)
            break;
        if (tries++ == 100) {
            /* give up after several tries */
            fprintf(stderr, "failed to sleep for 1 second!\n");
            return 1;
        }
    }
    uint64_t end = rdtsc_end();

    if (end < start || end - start < 1000000) {
        /* after a sleep of 1s, there must have been at least 1M ticks, otherwise smth wrong */
        fprintf(stderr, "RDTSC difference doesn't correspond to sleep of 1 second!\n");
        return 1;
    }

    printf("RDTSC: start = %lu, end = %lu, diff = %lu\n", start, end, end - start);
    puts("TEST OK!");
    return 0;
}
