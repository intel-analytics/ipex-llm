#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define TEST_DIR  "tmp"
#define TEST_FILE "__testfile__"

/* return true if file_name exists, otherwise false */
static bool find_file(const char* dir_name, const char* file_name) {
    bool found = false;
    DIR* dir = opendir(dir_name);
    if (dir == NULL) {
        perror("opendir failed");
        return found;
    }

    struct dirent* dent = NULL;
    while (1) {
        errno = 0;
        dent = readdir(dir);
        if (dent == NULL) {
            if (errno == 0)
                break;
            perror("readdir failed");
            goto out;
        }

        if (strncmp(file_name, dent->d_name, strlen(file_name)) == 0) {
            found = true;
            break;
        }
    }

out:
    closedir(dir);
    return found;
}

int main(int argc, const char** argv) {
    int rv = 0;
    int fd = 0;

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

    /* test readdir: should not find a file that we just deleted */
    if (find_file(TEST_DIR, TEST_FILE)) {
        perror("file " TEST_FILE " was unexpectedly found");
        return 1;
    }

    fd = open(TEST_DIR "/" TEST_FILE, O_CREAT | O_RDWR | O_EXCL,
              S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    if (fd < 0) {
        perror("open failed");
        return 1;
    }

    if (!find_file(TEST_DIR, TEST_FILE)) {
        perror("file " TEST_FILE " was not found");
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
    return rv;
}
