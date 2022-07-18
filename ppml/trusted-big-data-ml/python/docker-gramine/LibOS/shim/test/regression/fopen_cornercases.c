#define _XOPEN_SOURCE 700
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define FILENAME_MAX_LENGTH 255
#define PATH                "tmp/"
#define MSG                 "Hello World"

int main(int argc, char** argv) {
    size_t rets;
    int reti;
    char filename[FILENAME_MAX_LENGTH];
    memset(filename, 'a', sizeof(filename));
    filename[FILENAME_MAX_LENGTH - 1] = '\0';

    char filepath[sizeof(PATH) + sizeof(filename) - 1];
    strcpy(filepath, PATH);
    strcat(filepath, filename);

    printf("filepath = %s (len = %lu)\n", filepath, strlen(filepath));

    /* sanity check: try fopening empty filename (must fail) */
    FILE* fp = fopen("", "r");
    if (fp != NULL || errno != ENOENT) {
        perror("(sanity check) fopen with empty filename did not fail with ENOENT");
        return 1;
    }

    /* sanity check: try fopening dir in write mode (must fail) */
    fp = fopen(PATH, "w");
    if (fp != NULL || errno != EISDIR) {
        perror("(sanity check) fopen of dir with write access did not fail");
        return 1;
    }

    /* creating file within Gramine */
    int fd = openat(AT_FDCWD, "tmp/filecreatedbygramine", O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
        perror("failed to create file from within Gramine");
        return 1;
    }
    reti = close(fd);
    if (reti < 0) {
        perror("fclose failed");
        return 1;
    }

    /* write to file */
    fp = fopen(filepath, "w");
    if (fp == NULL) {
        perror("fopen failed");
        return 1;
    }

    rets = fwrite(MSG, sizeof(MSG), 1, fp); /* with NULL byte for later printf */
    if (rets != 1) {
        perror("fwrite failed");
        return 1;
    }

    reti = fclose(fp);
    if (reti) {
        perror("fclose failed");
        return 1;
    }

    /* read from same file */
    fp = fopen(filepath, "r");
    if (fp == NULL) {
        perror("fopen failed");
        return 1;
    }

    char buf[256];
    rets = fread(buf, 1, sizeof(buf), fp);
    if (rets != sizeof(MSG)) {
        perror("fread failed");
        return 1;
    }

    reti = fclose(fp);
    if (reti) {
        perror("fclose failed");
        return 1;
    }

    reti = printf("Successfully read from file: %s\n", buf);
    if (reti < 0) {
        perror("printf failed");
        return 1;
    }

    return 0;
}
