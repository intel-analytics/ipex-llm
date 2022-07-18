#define _XOPEN_SOURCE 700
#include <err.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

uint8_t* sig_stack;
size_t sig_stack_size = SIGSTKSZ;
_Atomic int count     = 0;

static void handler(int signal, siginfo_t* info, void* ucontext) {
    int ret;
    count++;

    uint8_t a;
    printf("sig %d count %d goes off with sp=%p, sig_stack: [%p, %p)\n", signal, count, &a,
           sig_stack, sig_stack + sig_stack_size);
    if (sig_stack <= &a && &a < sig_stack + sig_stack_size) {
        printf("OK on signal stack\n");
    } else {
        printf("FAIL out of signal stack\n");
    }
    fflush(stdout);

    stack_t old;
    memset(&old, 0, sizeof(old));
    ret = sigaltstack(NULL, &old);
    if (ret < 0) {
        err(EXIT_FAILURE, "sigaltstack in handler");
    }
    if (old.ss_flags & SS_ONSTACK) {
        printf("OK on sigaltstack in handler\n");
    } else {
        printf("FAIL on sigaltstack in handler\n");
    }

    /*
     * raise SIGALRM during signal handling to test nested signals
     * (three-levels deep nesting just to be sure)
     */
    if (count <= 2) {
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, SIGALRM);
        ret = sigprocmask(SIG_UNBLOCK, &set, NULL);
        if (ret) {
            err(EXIT_FAILURE, "sigprocmask");
        }
        raise(SIGALRM);
    }
    count--;
}

int main(int argc, char** argv) {
    int ret;
    sig_stack = malloc(sig_stack_size);
    if (sig_stack == NULL) {
        err(EXIT_FAILURE, "malloc");
    }

    stack_t ss = {
        .ss_sp    = sig_stack,
        .ss_flags = 0,
        .ss_size  = sig_stack_size,
    };
    stack_t old;
    memset(&old, 0xff, sizeof(old));
    ret = sigaltstack(&ss, &old);
    if (ret < 0) {
        err(EXIT_FAILURE, "sigaltstack");
    }
    if (old.ss_flags & SS_ONSTACK) {
        printf("FAIL on sigaltstack in main thread before alarm\n");
    } else {
        printf("OK on sigaltstack in main thread before alarm\n");
    }

    struct sigaction act;
    act.sa_sigaction = handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_SIGINFO | SA_NODEFER | SA_ONSTACK;
    ret = sigaction(SIGALRM, &act, NULL);
    if (ret < 0) {
        err(EXIT_FAILURE, "sigaction");
    }

    printf("&act == %p\n", &act);
    fflush(stdout);
    alarm(1);
    pause();

    memset(&old, 0xff, sizeof(old));
    ret = sigaltstack(NULL, &old);
    if (ret < 0) {
        err(EXIT_FAILURE, "sigaltstack");
    }
    if (old.ss_flags & SS_ONSTACK) {
        printf("FAIL on sigaltstack in main thread\n");
    } else {
        printf("OK on sigaltstack in main thread\n");
    }

    printf("done exiting\n");
    fflush(stdout);
    return 0;
}
