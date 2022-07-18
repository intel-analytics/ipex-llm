#define _GNU_SOURCE
#include <err.h>
#include <grp.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define ARRAY_LEN(x) (sizeof(x) / sizeof(x[0]))

static gid_t test_groups[] = { 0, 1337, 1337, 0 };

int main(void) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    int x = getgroups(0, NULL);
    if (x < 0) {
        err(1, "getgroups");
    }

    gid_t groups[ARRAY_LEN(test_groups)];
    memcpy(groups, test_groups, sizeof(groups));

    x = setgroups(ARRAY_LEN(groups), groups);
    if (x != 0) {
        err(1, "setgroups");
    }

    memset(groups, 0, sizeof(groups));

    pid_t pid = fork();
    if (pid < 0) {
        err(1, "fork");
    } else if (pid > 0) {
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            err(1, "waitpid");
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            errx(1, "invalid child return status: %d", status);
        }
    }

    x = getgroups(ARRAY_LEN(groups), groups);
    if (x < 0) {
        err(1, "getgroups");
    } else if (x != ARRAY_LEN(groups)) {
        errx(1, "getgroups returned invalid length: %d (expected: %zu)", x, ARRAY_LEN(groups));
    }

    for (size_t i = 0; i < ARRAY_LEN(groups); i++) {
        if (groups[i] != test_groups[i]) {
            errx(1, "invalid group: %d (expected: %d)", groups[i], test_groups[i]);
        }
    }

    printf("%s OK\n", pid == 0 ? "child" : "parent");
    return 0;
}

