#define _GNU_SOURCE
#include <dirent.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static unsigned int FILES_NO = 10000;

static int is_dot_or_dotdot(const char* name) {
    return (name[0] == '.' && !name[1]) || (name[0] == '.' && name[1] == '.' && !name[2]);
}

int main(int argc, char* argv[]) {
    int fd = 0, ret = 1;
    char name[21]       = {0};
    DIR* dir            = NULL;
    struct dirent* dent = NULL;
    unsigned long i     = 0;
    char* tmp_name      = NULL;
    char* old_wd        = NULL;
    unsigned char* seen = NULL;

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    if (argc != 2 && argc != 3) {
        fprintf(stderr, "Usage: %s tmp_folder_name [files_count]\n", argv[0]);
        return 1;
    }

    tmp_name = argv[1];

    if (argc > 2) {
        FILES_NO = atol(argv[2]);
    }

    if (!(seen = calloc(FILES_NO, sizeof *seen))) {
        err(1, "calloc");
    }

    if ((old_wd = get_current_dir_name()) == NULL) {
        err(1, "getcwd");
    }

    if (mkdir(tmp_name, S_IRWXU | S_IRWXG | S_IRWXO) < 0 || chdir(tmp_name) < 0) {
        err(1, "mkdir & chdir");
    }

    for (i = 0; i < FILES_NO; i++) {
        snprintf(name, sizeof(name), "%010lu", i);
        fd = open(name, O_CREAT | O_EXCL | O_RDWR, S_IRWXU | S_IRWXG | S_IRWXO);
        if (fd < 0) {
            fprintf(stderr, "error: cannot create file %lu: %s\n", i, strerror(errno));
            goto cleanup;
        }
        if (close(fd) < 0) {
            fprintf(stderr, "error: close failed with: %s\n", strerror(errno));
            goto cleanup;
        }
    }

    dir = opendir(".");
    if (!dir) {
        fprintf(stderr, "error: cannot open \".\": %s\n", strerror(errno));
        goto cleanup;
    }

    while (1) {
        errno = 0;
        dent  = readdir(dir);
        if (!dent) {
            if (errno != 0) {
                fprintf(stderr, "error: readdir: %s\n", strerror(errno));
                goto cleanup;
            } else {
                break;
            }
        }
        if (is_dot_or_dotdot(dent->d_name)) {
            continue;
        }
        i = atol(dent->d_name);
        if (i >= FILES_NO) {
            fprintf(stderr, "error: weird file found: \"%s\"\n", dent->d_name);
            goto cleanup;
        }
        if (seen[i]) {
            fprintf(stderr, "error: file \"%s\" seen multiple times\n", dent->d_name);
            goto cleanup;
        }
        seen[i] = 1;
    }

    for (i = 0; i < FILES_NO; i++) {
        if (!seen[i]) {
            fprintf(stderr, "error: file %lu not seen!\n", i);
            goto cleanup;
        }
    }

    puts("Success!");
    ret = 0;

cleanup:
    if (dir) {
        closedir(dir);
    }

    for (i = 0; i < FILES_NO; i++) {
        sprintf(name, "%010lu", i);
        if (unlink(name)) {
            ret = 1;
            fprintf(stderr, "error: could not remove file %s: %s\n", name, strerror(errno));
        }
    }

    if (chdir(old_wd) < 0) {
        ret = 1;
        fprintf(stderr, "error: could not change directory to original (%s): %s\n", old_wd,
                strerror(errno));
    }
    free(old_wd);

    if (rmdir(tmp_name) < 0) {
        ret = 1;
        fprintf(stderr, "error: could not remove tmp directory (%s): %s\n", tmp_name,
                strerror(errno));
    }

    return ret;
}
