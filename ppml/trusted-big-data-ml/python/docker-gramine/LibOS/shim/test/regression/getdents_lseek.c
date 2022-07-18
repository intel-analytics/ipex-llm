#define _GNU_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

struct linux_dirent64 {
    uint64_t d_ino;
    int64_t d_off;
    unsigned short d_reclen;
    unsigned char d_type;
    char d_name[];
};

#define BUF_SIZE 32

static int list_dir(const char* test_name, int fd) {
    char buf[BUF_SIZE];
    int count = 0;

    do {
        count = syscall(SYS_getdents64, fd, buf, BUF_SIZE);
        if (count < 0) {
            perror(test_name);
            return -1;
        }

        for (size_t offs = 0; offs < (size_t)count;) {
            struct linux_dirent64* d64 = (struct linux_dirent64*)(buf + offs);
            printf("%s: %s\n", test_name, d64->d_name);
            offs += d64->d_reclen;
        }
    } while (count > 0);
    return 0;
}

static int create_file(const char* dir, int n, mode_t perm) {
    int ret;
    char name[32];
    snprintf(name, sizeof(name), "%s/file%d", dir, n);
    ret = creat(name, perm);
    if (ret < 0) {
        perror("creat");
        return ret;
    }
    ret = close(ret);
    if (ret < 0) {
        perror("close");
        return ret;
    }
    return 0;
}

static int remove_file(const char* dir, int n) {
    int ret;
    char name[32];
    snprintf(name, sizeof(name), "%s/file%d", dir, n);
    ret = unlink(name);
    if (ret < 0) {
        perror("unlink");
        return ret;
    }
    return 0;
}

int main(void) {
    int ret, fd;
    const mode_t perm = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
    const int N = 5;

    ret = mkdir("root", perm);
    if (ret < 0) {
        perror("mkdir 1");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        ret = create_file("root", i, perm);
        if (ret < 0)
            return 1;
    }
    printf("getdents_lseek: setup ok\n");

    fd = open("root", O_RDONLY | O_DIRECTORY);
    if (fd < 0) {
        perror("open dir");
        return 1;
    }

    ret = list_dir("getdents64 1", fd);
    if (ret < 0)
        return 1;

    printf("getdents_lseek: removing file0 and creating file%d\n", N);

    ret = remove_file("root", 0);
    if (ret < 0)
        return 1;
    ret = create_file("root", N, perm);
    if (ret < 0)
        return 1;

    printf("getdents_lseek: seeking to 0\n");

    ret = lseek(fd, 0, SEEK_SET);
    if (ret < 0) {
        perror("lseek 1");
        return 1;
    }

    ret = list_dir("getdents64 2", fd);
    if (ret < 0)
        return 1;

    ret = close(fd);
    if (ret < 0) {
        perror("close dir");
        return 1;
    }

    // cleanup
    for (int i = 1; i < N + 1; i++) {
        ret = remove_file("root", i);
        if (ret < 0)
            return 1;
    }
    ret = rmdir("root");
    if (ret) {
        perror("rmdir");
        return 1;
    }
    return 0;
}
