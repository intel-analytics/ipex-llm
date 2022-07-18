#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdnoreturn.h>
#include <unistd.h>

static char* g_argv0 = NULL;
static char** g_argv = NULL;

static noreturn void do_exec(void) {
    execv(g_argv0, g_argv);
    perror("execve failed");
    exit(1);
}

static void* thread_func(void* arg) {
    do_exec();
    return arg;
}

int main(int argc, char** argv) {
    if (argc <= 0) {
        return 1;
    } else if (argc == 1) {
        return 0;
    }

    puts(argv[1]);
    fflush(stdout);

    argv[1] = argv[0];
    g_argv0 = argv[0];
    g_argv = &argv[1];

    /*
     * XXX: THIS PART OF TEST IS CURRENTLY DISABLED!!!
     * Unfortunately there is a bug somewhere, that sometimes causes some data race: the hash of
     * the first chunk of the executable file ('exec_same') is different from what's expected, even
     * though the hashed data matches. The function that does this hashing is: `load_trusted_file`.
     * This bug existed even before changes from PR introducing this test case.
     */
    (void)thread_func; // to satisfy compiler
    /* Creating another thread and doing a race on execve. Only one thread should survive. */
    // pthread_t th;
    // if (pthread_create(&th, NULL, thread_func, NULL) != 0) {
    //     perror("pthread_create failed");
    //     return 1;
    // }

    do_exec();
}
