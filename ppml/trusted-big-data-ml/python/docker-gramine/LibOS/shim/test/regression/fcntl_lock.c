/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/*
 * Test for POSIX locks (`fcntl(F_SETLK/F_SETLKW/F_GETLK`). We assert that the calls succeed (or
 * taking a lock fails), and log all details for debugging purposes.
 *
 * The tests usually start another process, and coordinate with it using pipes.
 */

#include <assert.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define TEST_DIR "tmp/"
#define TEST_FILE "tmp/lock_file"

static int g_fd;

static const char* str_cmd(int cmd) {
    switch (cmd) {
        case F_SETLK: return "F_SETLK";
        case F_SETLKW: return "F_SETLKW";
        case F_GETLK: return "F_GETLK";
        default: return "???";
    }
}

static const char* str_type(int type) {
    switch (type) {
        case F_RDLCK: return "F_RDLCK";
        case F_WRLCK: return "F_WRLCK";
        case F_UNLCK: return "F_UNLCK";
        default: return "???";
    }
}

static const char* str_whence(int whence) {
    switch (whence) {
        case SEEK_SET: return "SEEK_SET";
        case SEEK_CUR: return "SEEK_CUR";
        case SEEK_END: return "SEEK_END";
        default: return "???";
    }
}

static const char* str_err(int err) {
    switch (err) {
        case EACCES: return "EACCES";
        case EAGAIN: return "EAGAIN";
        default: return "???";
    }
}

/* Run fcntl command and log it, along with the result. Exit on unexpected errors. */
static int try_fcntl(int cmd, struct flock* fl) {
    /* Save the initial values before `fl` is modified, so that we can log them after the call */
    int type = fl->l_type;
    int whence = fl->l_whence;
    off_t start = fl->l_start;
    off_t len = fl->l_len;

    assert(cmd == F_SETLK || cmd == F_SETLKW || cmd == F_GETLK);
    assert(type == F_RDLCK || type == F_WRLCK || type == F_UNLCK);
    assert(whence == SEEK_SET || whence == SEEK_CUR || whence == SEEK_END);

    int ret = fcntl(g_fd, cmd, fl);
    fprintf(stderr, "%d: fcntl(fd, %s, {%s, %s, %4ld, %4ld}) = %d", getpid(), str_cmd(cmd),
            str_type(type), str_whence(whence), start, len, ret);
    if (ret == -1)
        fprintf(stderr, " (%s)", str_err(errno));
    if (ret == 0 && cmd == F_GETLK) {
        if (fl->l_type == F_UNLCK) {
            fprintf(stderr, "; {%s}\n", str_type(fl->l_type));
        } else {
            fprintf(stderr, "; {%s, %s, %4ld, %4ld, %d}\n", str_type(fl->l_type),
                    str_whence(fl->l_whence), fl->l_start, fl->l_len, fl->l_pid);
        }
    } else {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    if (ret != -1 && ret != 0)
        errx(1, "fcntl returned unexpected value");
    if (ret == -1) {
        /* We permit -1 only for F_SETLK, and only with EACCES or EAGAIN errors (which means the
         * lock could not be placed immediately). */
        if (!(cmd == F_SETLK && (errno == EACCES || errno == EAGAIN))) {
            err(1, "fcntl");
        }
    }
    return ret;
}

/* Wrapper for `try_fcntl`. Returns true if it succeeds (F_SETLK returns success, F_GETLK returns no
 * conflicting lock). */
static bool try_lock(int cmd, int type, int whence, long int start, long int len) {
    struct flock fl = {
        .l_type = type,
        .l_whence = whence,
        .l_start = start,
        .l_len = len,
    };
    int ret = try_fcntl(cmd, &fl);

    if (cmd == F_GETLK) {
        assert(ret == 0);

        return fl.l_type == F_UNLCK;
    } else {
        return ret == 0;
    }
}

/* Check whether F_GETLK returns the right conflicting lock. */
static void lock_check(int type, long int start, long int len, int conflict_type,
                       long int conflict_start, long int conflict_len) {
    assert(conflict_type != F_UNLCK);

    struct flock fl = {
        .l_type = type,
        .l_whence = SEEK_SET,
        .l_start = start,
        .l_len = len,
    };
    int ret = try_fcntl(F_GETLK, &fl);
    if (ret == -1)
        err(1, "fcntl");
    if (fl.l_type != conflict_type || fl.l_whence != SEEK_SET || fl.l_start != conflict_start
            || fl.l_len != conflict_len) {
        /* `try_fcntl()` already printed the actual result */
        errx(1, "F_GETLK returned wrong lock; expected {%s, SEEK_SET, %ld, %ld)",
             str_type(conflict_type), conflict_start, conflict_len);
    }
}

static void unlock(long int start, long int len) {
    if (!try_lock(F_SETLK, F_UNLCK, SEEK_SET, start, len))
        errx(1, "unlock failed");
}

static void lock(int type, long int start, long int len) {
    assert(type == F_RDLCK || type == F_WRLCK);

    if (!try_lock(F_GETLK, type, SEEK_SET, start, len)
            || !try_lock(F_SETLK, type, SEEK_SET, start, len))
        errx(1, "setting %s failed", str_type(type));
}

static void lock_wait_ok(int type, long int start, long int len) {
    if (!try_lock(F_SETLKW, type, SEEK_SET, start, len))
        errx(1, "waiting for %s failed", str_type(type));
}

static void lock_fail(int type, long int start, long int len) {
    if (try_lock(F_GETLK, type, SEEK_SET, start, len)
            || try_lock(F_SETLK, type, SEEK_SET, start, len))
        errx(1, "setting %s succeeded unexpectedly", str_type(type));
}

/*
 * Test: lock/unlock various ranges. The locks are all for the same process, so the test is unlikely
 * to fail, but it's useful for checking if the locks are replaced and merged correctly (by looking
 * at Gramine debug output).
 */
static void test_ranges(void) {
    printf("testing ranges...\n");
    unlock(0, 0);

    /* Lock some ranges, check joining adjacent ranges */
    lock(F_RDLCK, 10, 10);
    lock(F_RDLCK, 30, 10);
    lock(F_RDLCK, 20, 10);
    lock(F_RDLCK, 1000, 0);

    /* Unlock some ranges, check subtracting and splitting ranges */
    unlock(5, 10);
    unlock(20, 5);
    unlock(35, 10);
    unlock(950, 100);

    /* Overwrite with write lock */
    lock(F_WRLCK, 0, 30);
    lock(F_WRLCK, 30, 30);
}

static void wait_for_child(void) {
    int ret;
    do {
        ret = wait(NULL);
    } while (ret == -1 && errno == EINTR);
    if (ret == -1)
        err(1, "wait");
}

static void open_pipes(int pipes[2][2]) {
    for (unsigned int i = 0; i < 2; i++) {
        if (pipe(pipes[i]) < 0)
            err(1, "pipe");
    }
}

static void close_pipes(int pipes[2][2]) {
    for (unsigned int i = 0; i < 2; i++) {
        for (unsigned int j = 0; j < 2; j++) {
            if (close(pipes[i][j]) < 0)
                err(1, "close pipe");
        }
    }
}

static void write_pipe(int pipe[2]) {
    char c = 0;
    int ret;
    do {
        ret = write(pipe[1], &c, sizeof(c));
    } while (ret == -1 && errno == EINTR);
    if (ret == -1)
        err(1, "write");
}

static void read_pipe(int pipe[2]) {
    char c;
    int ret;
    do {
        ret = read(pipe[0], &c, sizeof(c));
    } while (ret == -1 && errno == EINTR);
    if (ret == -1)
        err(1, "read");
    if (ret == 0)
        errx(1, "pipe closed");
}

/* Test: child takes a lock and then exits. The lock should be released. */
static void test_child_exit(void) {
    printf("testing child exit...\n");
    unlock(0, 0);

    int pipes[2][2];
    open_pipes(pipes);

    pid_t pid = fork();
    if (pid < 0)
        err(1, "fork");

    if (pid == 0) {
        lock(F_WRLCK, 0, 100);
        write_pipe(pipes[0]);
        read_pipe(pipes[1]);
        exit(0);
    }

    read_pipe(pipes[0]);
    lock_fail(F_RDLCK, 0, 100);
    write_pipe(pipes[1]);
    lock_wait_ok(F_RDLCK, 0, 100);

    wait_for_child();
    close_pipes(pipes);
}

/* Test: child takes a lock, and then closes a duplicated FD. The lock should be released. */
static void test_file_close(void) {
    printf("testing file close...\n");
    unlock(0, 0);

    int pipes[2][2];
    open_pipes(pipes);

    pid_t pid = fork();
    if (pid < 0)
        err(1, "fork");

    if (pid == 0) {
        lock(F_WRLCK, 0, 100);
        write_pipe(pipes[0]);
        read_pipe(pipes[1]);

        int fd2 = dup(g_fd);
        if (fd2 < 0)
            err(1, "fopen");

        if (close(fd2) < 0)
            err(1, "close");

        read_pipe(pipes[1]);
        exit(0);
    }

    read_pipe(pipes[0]);
    lock_fail(F_RDLCK, 0, 100);
    write_pipe(pipes[1]);
    lock_wait_ok(F_RDLCK, 0, 100);
    write_pipe(pipes[1]);

    wait_for_child();
    close_pipes(pipes);
}

/* Test: child waits for parent to release a lock. */
static void test_child_wait(void) {
    printf("testing child wait...\n");
    unlock(0, 0);

    int pipes[2][2];
    open_pipes(pipes);

    lock(F_RDLCK, 0, 100);

    pid_t pid = fork();
    if (pid < 0)
        err(1, "fork");

    if (pid == 0) {
        lock(F_RDLCK, 0, 100);
        lock_fail(F_WRLCK, 0, 100);
        write_pipe(pipes[0]);
        lock_wait_ok(F_WRLCK, 0, 100);
        exit(0);
    }

    read_pipe(pipes[0]);
    unlock(0, 100);

    wait_for_child();
    close_pipes(pipes);
}

/* Test: parent waits for child to release a lock. */
static void test_parent_wait(void) {
    printf("testing parent wait...\n");
    unlock(0, 0);

    int pipes[2][2];
    open_pipes(pipes);

    pid_t pid = fork();
    if (pid < 0)
        err(1, "fork");

    if (pid == 0) {
        lock(F_RDLCK, 0, 100);
        write_pipe(pipes[0]);
        read_pipe(pipes[1]);
        unlock(0, 100);
        read_pipe(pipes[1]);
        exit(0);
    }

    /* parent process: */

    read_pipe(pipes[0]);

    /* read lock should succeed */
    lock(F_RDLCK, 0, 100);
    lock_fail(F_WRLCK, 0, 100);
    write_pipe(pipes[1]);
    lock_wait_ok(F_WRLCK, 0, 100);
    write_pipe(pipes[1]);

    wait_for_child();
    close_pipes(pipes);
}

/* Test: check that a range until EOF (len == 0) is handled correctly. */
static void test_range_with_eof(void) {
    printf("testing range with EOF...\n");
    unlock(0, 0);

    int pipes[2][2];
    open_pipes(pipes);

    pid_t pid = fork();
    if (pid < 0)
        err(1, "fork");

    if (pid == 0) {
        /* lock [100 .. EOF] */
        lock(F_WRLCK, 100, 0);
        write_pipe(pipes[0]);
        read_pipe(pipes[1]);
        exit(0);
    }

    read_pipe(pipes[0]);
    /* check [50 .. 149], we should see a conflicting lock for [100 .. EOF] */
    lock_check(F_WRLCK, 50, 100, F_WRLCK, 100, 0);
    write_pipe(pipes[1]);

    wait_for_child();
    close_pipes(pipes);
}


int main(void) {
    setbuf(stdout, NULL);

    g_fd = open(TEST_FILE, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (g_fd < 0)
        err(1, "open");

    test_ranges();
    test_child_exit();
    test_file_close();
    test_child_wait();
    test_parent_wait();
    test_range_with_eof();

    if (close(g_fd) < 0)
        err(1, "close");

    if (unlink(TEST_FILE) < 0)
        err(1, "unlink");

    printf("TEST OK\n");
    return 0;
}
