#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "rw_file.h"

#define STR_HELPER(x) #x
#define STR(x)        STR_HELPER(x)

#define BUF_LENGTH  4096
#define HOLE_LENGTH 8192

#define TEST_DIR  "tmp"
#define TEST_FILE "__testfile__"

int main(int argc, const char** argv) {
    char buf[BUF_LENGTH];
    ssize_t bytes;
    int fd = 0;
    int rv = 0;

    memset(buf, 0, BUF_LENGTH);

    /* setup the test directory and file */
    rv = mkdir(TEST_DIR, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (rv < 0 && errno != EEXIST) {
        perror("mkdir failed");
        return 1;
    }

    rv = unlink(TEST_DIR "/" TEST_FILE);
    if (rv < 0 && errno != ENOENT) {
        perror("unlink failed");
        return 1;
    }

    fd = open(TEST_DIR "/" TEST_FILE, O_CREAT | O_RDWR | O_EXCL,
              S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    if (fd < 0) {
        perror("open failed");
        return 1;
    }

    /* test file size: write a file of type != FILEBUF_MAP */
    bytes = posix_fd_write(fd, buf, BUF_LENGTH);
    if (bytes != BUF_LENGTH) {
        perror("writing " STR(BUF_LENGTH) " bytes to test file failed");
        return 1;
    }

    rv = lseek(fd, 0, SEEK_SET);
    if (rv < 0) {
        perror("lseek to beginning of file failed");
        return 1;
    }

    bytes = posix_fd_write(fd, buf, 1);
    if (bytes != 1) {
        perror("writing one byte to test file failed");
        return 1;
    }

    rv = lseek(fd, HOLE_LENGTH, SEEK_SET);
    if (rv < 0) {
        perror("lseek to " STR(HOLE_LENGTH) " offset failed");
        return 1;
    }

    bytes = posix_fd_write(fd, buf, BUF_LENGTH);
    if (bytes != BUF_LENGTH) {
        perror("writing " STR(BUF_LENGTH) " bytes to test file failed");
        return 1;
    }

    rv = close(fd);
    if (rv < 0) {
        perror("close failed");
        return 1;
    }

    fd = open(TEST_DIR "/" TEST_FILE, O_RDONLY);
    if (fd < 0) {
        perror("open failed");
        return 1;
    }

    /* reopen file: file size should be HOLE_LENGTH + BUF_LENGTH */
    rv = lseek(fd, HOLE_LENGTH, SEEK_SET);
    if (rv < 0) {
        perror("lseek to " STR(HOLE_LENGTH) " offset failed");
        return 1;
    }

    bytes = posix_fd_read(fd, buf, BUF_LENGTH);
    if (bytes != BUF_LENGTH) {
        perror("reading " STR(BUF_LENGTH) " bytes from test file failed");
        return 1;
    }

    rv = close(fd);
    if (rv < 0) {
        perror("close failed");
        return 1;
    }

    rv = unlink(TEST_DIR "/" TEST_FILE);
    if (rv < 0) {
        perror("unlink failed");
        return 1;
    }

    printf("test completed successfully\n");
    return 0;
}
