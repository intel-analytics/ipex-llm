#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAX_EFDS 3

#define CLOSE_EFD_AND_EXIT(efd) \
    do {                        \
        close(efd);             \
        return 1;               \
    } while (0)

#define EXIT_IF_ERROR(efd, bytes, prefix)           \
    do {                                            \
        if ((bytes) != sizeof(uint64_t)) {          \
            perror(prefix);                         \
            printf("error at line %d\n", __LINE__); \
            CLOSE_EFD_AND_EXIT(efd);                \
        }                                           \
    } while (0)

int efds[MAX_EFDS] = {0};

static void* write_eventfd_thread(void* arg) {
    uint64_t count = 10;

    int* efds = (int*)arg;

    if (!arg) {
        printf("arg is NULL\n");
        return NULL;
    }

    for (int i = 0; i < MAX_EFDS; i++) {
        printf("%s: efd: %d\n", __func__, efds[i]);
    }

    for (int i = 0; i < MAX_EFDS; i++) {
        sleep(1);
        if (write(efds[i], &count, sizeof(count)) != sizeof(count)) {
            perror("write error");
            return NULL;
        }
        count += 1;
    }

    return NULL;
}

/* This function used to test polling on a group of eventfd descriptors.
 * To support regression testing, positive value returned for error case. */
static int eventfd_using_poll(void) {
    int ret = 0;
    struct pollfd pollfds[MAX_EFDS];
    pthread_t tid = 0;
    uint64_t count = 0;
    int poll_ret = 0;
    int nread_events = 0;

    for (int i = 0; i < MAX_EFDS; i++) {
        efds[i] = eventfd(0, 0);

        if (efds[i] < 0) {
            perror("eventfd failed");
            return 1;
        }

        printf("efd = %d\n", efds[i]);

        pollfds[i].fd     = efds[i];
        pollfds[i].events = POLLIN;
    }

    ret = pthread_create(&tid, NULL, write_eventfd_thread, efds);

    if (ret != 0) {
        perror("error in thread creation");
        ret = 1;
        goto out;
    }

    while (1) {
        poll_ret = poll(pollfds, MAX_EFDS, 5000);

        if (poll_ret == 0) {
            printf("Poll timed out. Exiting.\n");
            break;
        }

        if (poll_ret < 0) {
            perror("error from poll");
            ret = 1;
            break;
        }

        for (int i = 0; i < MAX_EFDS; i++) {
            if (pollfds[i].revents & POLLIN) {
                pollfds[i].revents = 0;
                errno = 0;
                if (read(pollfds[i].fd, &count, sizeof(count)) != sizeof(count)) {
                    perror("read error");
                    ret = 1;
                    goto out;
                }
                printf("fd set: %d\n", pollfds[i].fd);
                printf("efd: %d, count: %lu, errno: %d\n", pollfds[i].fd, count, errno);
                nread_events++;
            }
        }
    }

    if (nread_events == MAX_EFDS)
        printf("%s completed successfully\n", __func__);
    else
        printf("%s: nread_events: %d, MAX_EFDS: %d\n", __func__, nread_events, MAX_EFDS);

    pthread_join(tid, NULL);

out:
    for (int i = 0; i < MAX_EFDS; i++) {
        close(efds[i]);
    }

    return ret;
}

/* This function used to test various flags supported while creating eventfd
 * descriptors.
 * To support regression testing, positive value returned for error case. */
static int eventfd_using_various_flags(void) {
    uint64_t count = 0;
    int efd = 0;
    ssize_t bytes = 0;
    int eventfd_flags[] = {0, EFD_SEMAPHORE, EFD_NONBLOCK, EFD_CLOEXEC};

    for (unsigned int i = 0; i < sizeof(eventfd_flags) / sizeof(*eventfd_flags); i++) {
        printf("iteration %d, flags %d\n", i, eventfd_flags[i]);

        efd = eventfd(0, eventfd_flags[i]);

        if (efd < 0) {
            perror("eventfd failed");
            printf("eventfd error for iteration %d, flags %d\n", i, eventfd_flags[i]);
            return 1;
        }

        count = 5;
        bytes = write(efd, &count, sizeof(count));
        EXIT_IF_ERROR(efd, bytes, "write");

        bytes = write(efd, &count, sizeof(count));
        EXIT_IF_ERROR(efd, bytes, "write");

        count = 0;
        errno = 0;
        if (eventfd_flags[i] & EFD_SEMAPHORE) {
            uint64_t prev_count = 0;
            bytes = read(efd, &prev_count, sizeof(prev_count));
            EXIT_IF_ERROR(efd, bytes, "read");

            bytes = read(efd, &count, sizeof(count));
            EXIT_IF_ERROR(efd, bytes, "read");

            if (prev_count != 1 || count != 1) {
                printf("flag->EFD_SEMAPHORE, error, prev_count: %lu, new count: %lu\n", prev_count,
                       count);
                close(efd);
                return 1;
            }
            close(efd);
            continue;
        }

        count = 0;
        errno = 0;
        bytes = read(efd, &count, sizeof(count));
        EXIT_IF_ERROR(efd, bytes, "read");
        if (count != 10) {
            printf("%d: efd: %d, count: %lu, errno: %d\n", __LINE__, efd, count, errno);
            CLOSE_EFD_AND_EXIT(efd);
        }

        /* calling the second read would block if flags doesn't have EFD_NONBLOCK */
        if (eventfd_flags[i] & EFD_NONBLOCK) {
            count = 0;
            errno = 0;
            ssize_t ret = read(efd, &count, sizeof(count));
            if (ret != -1 || errno != EAGAIN) {
                printf("read that should return -1 with EAGAIN returned %ld with errno %d\n", ret,
                       errno);
                close(efd);
                return 1;
            }
            printf("%d: efd: %d, count: %lu, errno: %d\n", __LINE__, efd, count, errno);
        }

        close(efd);
    }

    printf("%s completed successfully\n", __func__);

    return 0;
}

static int eventfd_using_fork(void) {
    int status     = 0;
    int efd        = 0;
    uint64_t count = 0;

    efd = eventfd(0, EFD_NONBLOCK);

    if (efd < 0) {
        perror("eventfd failed");
        return 1;
    }

    pid_t pid = fork();

    if (pid == 0) {
        // child process
        count = 5;
        if (write(efd, &count, sizeof(count)) != sizeof(count)) {
            perror("write error");
            exit(1);
        }
        exit(0);
    } else if (pid > 0) {
        // parent process
        waitpid(pid, &status, 0);

        if (WIFSIGNALED(status)) {
            perror("child was terminated by signal");
            CLOSE_EFD_AND_EXIT(efd);
        }

        count = 0;
        if (read(efd, &count, sizeof(count)) != sizeof(count)) {
            perror("read error");
            close(efd);
            exit(1);
        }
        if (count != 5) {
            printf("parent-pid: %d, efd: %d, count: %lu, errno: %d\n", getpid(), efd, count, errno);
            CLOSE_EFD_AND_EXIT(efd);
        }

    } else {
        perror("fork error");
        CLOSE_EFD_AND_EXIT(efd);
    }

    close(efd);

    printf("%s completed successfully\n", __func__);

    return 0;
}

int main(int argc, char* argv[]) {
    int ret;

    ret = eventfd_using_poll();
    if (ret) {
        puts("eventfd_using_poll() failed");
        return 1;
    }
    puts("----------------------------------------");
    ret = eventfd_using_various_flags();
    if (ret) {
        puts("eventfd_using_various_flags() failed");
        return 1;
    }
    puts("----------------------------------------");
    ret = eventfd_using_fork();
    if (ret) {
        puts("eventfd_using_fork() failed");
        return 1;
    }

    return 0;
}
