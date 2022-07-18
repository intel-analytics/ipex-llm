#define _XOPEN_SOURCE 700
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x)        \
    do {                \
        if (x) {        \
            perror(#x); \
            exit(1);    \
        }               \
    } while (0)

static int seen_signal_cnt = 0;

static void signal_handler(int signal) {
    __atomic_add_fetch(&seen_signal_cnt, 1, __ATOMIC_RELAXED);
    printf("signal handled: %d\n", signal);
}

static void ignore_signal(int sig) {
    sigset_t newmask;
    sigemptyset(&newmask);
    if (sig) {
        sigaddset(&newmask, sig);
    }

    CHECK(sigprocmask(SIG_SETMASK, &newmask, NULL) < 0);
}

static void set_signal_handler(int sig, void* handler) {
    struct sigaction act = {
        .sa_handler = handler,
    };
    CHECK(sigaction(sig, &act, NULL) < 0);
}

static void test_sigprocmask(void) {
    sigset_t newmask;
    sigset_t oldmask;
    sigemptyset(&newmask);
    sigemptyset(&oldmask);
    sigaddset(&newmask, SIGKILL);
    sigaddset(&newmask, SIGSTOP);

    CHECK(sigprocmask(SIG_SETMASK, &newmask, NULL) < 0);

    CHECK(sigprocmask(SIG_SETMASK, NULL, &oldmask) < 0);

    if (sigismember(&oldmask, SIGKILL) || sigismember(&oldmask, SIGSTOP)) {
        printf("SIGKILL or SIGSTOP should be ignored, but is not.\n");
        exit(1);
    }
}

static void clean_mask_and_pending_signals(void) {
    /* We should not have any pending signals other than SIGALRM. */
    set_signal_handler(SIGALRM, signal_handler);
    /* This assumes that unblocking a signal will cause its immediate delivery. */
    ignore_signal(0);
    __atomic_store_n(&seen_signal_cnt, 0, __ATOMIC_RELAXED);
}

static void test_multiple_pending(void) {
    ignore_signal(SIGALRM);

    set_signal_handler(SIGALRM, signal_handler);

    CHECK(kill(getpid(), SIGALRM) < 0);
    CHECK(kill(getpid(), SIGALRM) < 0);

    if (__atomic_load_n(&seen_signal_cnt, __ATOMIC_RELAXED) != 0) {
        printf("Handled a blocked standard signal!\n");
        exit(1);
    }

    ignore_signal(0);

    if (__atomic_load_n(&seen_signal_cnt, __ATOMIC_RELAXED) != 1) {
        printf("Multiple or none instances of standard signal were queued!\n");
        exit(1);
    }

    __atomic_store_n(&seen_signal_cnt, 0, __ATOMIC_RELAXED);

    int sig = SIGRTMIN;
    ignore_signal(sig);

    CHECK(kill(getpid(), sig) < 0);
    CHECK(kill(getpid(), sig) < 0);

    set_signal_handler(sig, signal_handler);

    if (__atomic_load_n(&seen_signal_cnt, __ATOMIC_RELAXED) != 0) {
        printf("Handled a blocked real-time signal!\n");
        exit(1);
    }

    ignore_signal(0);

    if (__atomic_load_n(&seen_signal_cnt, __ATOMIC_RELAXED) != 2) {
        printf("Multiple instances of real-time signal were NOT queued!\n");
        exit(1);
    }
}

static void test_fork(void) {
    ignore_signal(SIGALRM);

    set_signal_handler(SIGALRM, signal_handler);

    CHECK(kill(getpid(), SIGALRM) < 0);

    pid_t p = fork();
    CHECK(p < 0);
    if (p == 0) {
        ignore_signal(0);

        if (__atomic_load_n(&seen_signal_cnt, __ATOMIC_RELAXED) != 0) {
            printf("Pending signal was inherited after fork!\n");
            exit(1);
        }

        puts("Child OK");
        exit(0);
    }

    set_signal_handler(SIGALRM, SIG_DFL);

    CHECK(waitpid(p, NULL, 0) != p);
}

static void test_execve_start(char* self) {
    ignore_signal(SIGALRM);

    set_signal_handler(SIGALRM, SIG_DFL);

    CHECK(kill(getpid(), SIGALRM) < 0);

    char* argv[] = {self, (char*)"cont", NULL};
    CHECK(execve(self, argv, NULL));
}

static void test_execve_continue(void) {
    set_signal_handler(SIGALRM, signal_handler);

    if (__atomic_load_n(&seen_signal_cnt, __ATOMIC_RELAXED) != 0) {
        printf("Seen an unexpected signal!\n");
        exit(1);
    }

    ignore_signal(0);

    if (__atomic_load_n(&seen_signal_cnt, __ATOMIC_RELAXED) != 1) {
        printf("Pending signal was NOT preserved across execve!\n");
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 1) {
        return 1;
    } else if (argc > 1) {
        test_execve_continue();
        puts("All tests OK");
        return 0;
    }

    test_sigprocmask();

    clean_mask_and_pending_signals();
    test_multiple_pending();

    clean_mask_and_pending_signals();
    test_fork();

    clean_mask_and_pending_signals();
    test_execve_start(argv[0]);
    return 1;
}
