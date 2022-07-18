#define _GNU_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

struct linux_dirent {
    unsigned long d_ino;
    unsigned long d_off;
    unsigned short d_reclen;
    char d_name[];
};

struct linux_dirent64 {
    uint64_t d_ino;
    int64_t d_off;
    unsigned short d_reclen;
    unsigned char d_type;
    char d_name[];
};

#define BUF_SIZE 512
#define BUF_SIZE_SMALL 64

static int do_getdents(const char* name, int fd, size_t buf_size) {
    char buf[buf_size];

    int count = syscall(SYS_getdents, fd, buf, buf_size);
    if (count < 0) {
        perror(name);
        return -1;
    }
    for (size_t offs = 0; offs < (size_t)count;) {
        struct linux_dirent* d32 = (struct linux_dirent*)(buf + offs);
        char d_type = *(buf + offs + d32->d_reclen - 1);
        printf("%s: %s [0x%x]\n", name, d32->d_name, d_type);
        offs += d32->d_reclen;
    }
    return count;
}

static int do_getdents64(const char* name, int fd, size_t buf_size) {
    char buf[buf_size];

    int count = syscall(SYS_getdents64, fd, buf, buf_size);
    if (count < 0) {
        perror(name);
        return -1;
    }

    for (size_t offs = 0; offs < (size_t)count;) {
        struct linux_dirent64* d64 = (struct linux_dirent64*)(buf + offs);
        printf("%s: %s [0x%x]\n", name, d64->d_name, d64->d_type);
        offs += d64->d_reclen;
    }
    return count;
}

int main(void) {
    int rv, fd, count;
    const mode_t perm = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;

    // setup
    // We test a directory one level below root so ".." is also present in SGX.
    rv = mkdir("root", perm);
    if (rv) {
        perror("mkdir 1");
        return 1;
    }
    rv = mkdir("root/testdir", perm);
    if (rv) {
        perror("mkdir 2");
        return 1;
    }
    rv = creat("root/testdir/file1", perm);
    if (rv < 0) {
        perror("creat 1");
        return 1;
    }
    rv = close(rv);
    if (rv) {
        perror("close 1");
        return 1;
    }
    rv = creat("root/testdir/file2", perm);
    if (rv < 0) {
        perror("creat 2");
        return 1;
    }
    rv = close(rv);
    if (rv) {
        perror("close 2");
        return 1;
    }
    rv = mkdir("root/testdir/dir3", perm);
    if (rv) {
        perror("mkdir 3");
        return 1;
    }
    // enable symlink when implemented, or just use the LTP test
    // rv = symlink("root/testdir/file2", "root/testdir/link4");
    // if (rv) { perror ("symlink"); return 1; }
    printf("getdents: setup ok\n");

    // 32-bit listing
    fd = open("root/testdir", O_RDONLY | O_DIRECTORY);
    if (fd < 0) {
        perror("open 1");
        return 1;
    }

    do {
        count = do_getdents("getdents32", fd, BUF_SIZE);
        if (count < 0)
            return 1;
    } while (count > 0);

    rv = close(fd);
    if (rv) {
        perror("close 1");
        return 1;
    }

    // 64-bit listing
    fd = open("root/testdir", O_RDONLY | O_DIRECTORY);
    if (fd < 0) {
        perror("open 2");
        return 1;
    }

    do {
        count = do_getdents64("getdents64", fd, BUF_SIZE);
        if (count < 0)
            return 1;
    } while (count > 0);

    rv = close(fd);
    if (rv) {
        perror("close 2");
        return 1;
    }

    /*
     * Check if dir handle works across fork: call getdents64 once before fork, then once after fork
     * by both parent and child, with a buffer size small enough so that all three calls return
     * something (if the buffer is too big, the call before fork might retrieve all the entries, and
     * subsequent calls will return 0).
     *
     * Note that we currently do not synchronize handle state between parent and child in Gramine,
     * so both calls after fork will return the same entries.
     */
    fd = open("root/testdir", O_RDONLY | O_DIRECTORY);

    count = do_getdents64("getdents64 before fork", fd, BUF_SIZE_SMALL);
    if (count < 0)
        return 1;

    fflush(stdout);

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    const char* process = pid ? "parent getdents64" : "child getdents64";
    count = do_getdents64(process, fd, BUF_SIZE_SMALL);
    if (count < 0)
        return 1;
    if (count == 0) {
        fprintf(stderr, "%s: handle empty too soon\n", process);
        return 1;
    }

    rv = close(fd);
    if (rv) {
        perror("close after fork");
        return 1;
    }

    if (pid) {
        // wait for child
        int status;
        rv = waitpid(pid, &status, 0);
        if (rv < 0) {
            perror("waitpid");
            return 1;
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "waitpid: got %d\n", status);
            return 1;
        }

        // cleanup
        remove("root/testdir/file1");
        remove("root/testdir/file2");
        remove("root/testdir/dir3");
        remove("root/testdir");
        remove("root");
    }
    return 0;
}
