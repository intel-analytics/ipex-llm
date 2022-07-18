#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s read|write|append <filename>\n", argv[0]);
        return 1;
    }

    /* we use append instead of write simply to not overwrite the file */
    const char* mode;
    if (!strcmp(argv[1], "read")) {
        mode = "r";
    } else if (!strcmp(argv[1], "write")) {
        mode = "r+"; // without file creation
    } else if (!strcmp(argv[1], "append")) {
        mode = "a";
    } else {
        errx(1, "invalid cmdline argument");
    }
    FILE* fp = fopen(argv[2], mode);
    if (!fp) {
        perror("fopen failed");
        return 2;
    }

    int ret = fclose(fp);
    if (ret) {
        perror("fclose failed");
        return 3;
    }

    printf("file_check_policy succeeded\n");

    return 0;
}
