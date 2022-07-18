#define _XOPEN_SOURCE 700
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

/*
 * Test the process exit logic. Make Nth thread run exit_group(), which should override exit code of
 * the process.
 *
 * The above is done in a forked process, so that we also check if the right exit code propagates
 * through wait().
 */

#define THREAD_NUM 4

int exit_codes[THREAD_NUM];
pthread_barrier_t barrier;

static void* run(void* arg) {
    pthread_barrier_wait(&barrier);

    int* code = arg;
    if (*code != 0)
        exit(*code);

    // Wait on the barrier again - one of the processes will exit, so we shouldn't ever finish
    pthread_barrier_wait(&barrier);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s thread_idx exit_code\n", argv[0]);
        exit(255);
    }

    int thread_idx = atoi(argv[1]);
    int exit_code = atoi(argv[2]);

    if (!(0 <= thread_idx && thread_idx < THREAD_NUM)) {
        fprintf(stderr, "thread_idx should be between 0 and %d exclusive\n", THREAD_NUM);
        exit(255);
    }

    if (exit_code == 0) {
        fprintf(stderr, "exit_code should not be 0\n");
        exit(255);
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(255);
    } else if (pid > 0) {
        int status;
        if (wait(&status) < 0) {
            perror("wait");
            exit(255);
        }
        if (!WIFEXITED(status)) {
            fprintf(stderr, "wrong wait() status: %d\n", status);
            exit(255);
        }
        exit(WEXITSTATUS(status));
    }

    exit_codes[thread_idx] = exit_code;

    pthread_t thread[THREAD_NUM];
    pthread_barrier_init(&barrier, NULL, THREAD_NUM);

    for (int j = 1; j < THREAD_NUM; j++) {
        pthread_create(&thread[j], NULL, run, &exit_codes[j]);
    }

    run(&exit_codes[0]);
    return 0; // should not be reached
}
