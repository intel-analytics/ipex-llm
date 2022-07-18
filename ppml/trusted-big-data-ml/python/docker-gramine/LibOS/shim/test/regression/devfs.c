#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char** argv) {
    int ret;
    size_t n;
    FILE* f;
    DIR* dir;
    struct dirent* dirent;
    char buf[64];

    /* sanity checks that incorrect invocations return errors */
    f = fopen("/dummy-pseudo-fs", "r");
    if (f != NULL || errno != ENOENT) {
        perror("(sanity check) fopen of dummy non-existing pseudo-FS did not fail with ENOENT");
        return 1;
    }

    f = fopen("/dev", "r+");
    if (f != NULL || errno != EISDIR) {
        perror("(sanity check) fopen of /dev did not fail with EISDIR");
        return 1;
    }

    f = fopen("/dev/dummy", "r");
    if (f != NULL || errno != ENOENT) {
        perror("(sanity check) fopen of /dev/dummy (non-existing file) did not fail with ENOENT");
        return 1;
    }

    printf("===== Contents of /dev\n");
    dir = opendir("/dev");
    if (!dir) {
        perror("opendir /dev");
        return 1;
    }

    errno = 0;
    while ((dirent = readdir(dir))) {
        printf("/dev/%s\n", dirent->d_name);
    }
    if (errno) {
        perror("readdir /dev");
        return 1;
    }

    ret = closedir(dir);
    if (ret < 0) {
        perror("closedir /dev");
        return 1;
    }

    printf("===== Read from /dev/urandom\n");
    f = fopen("/dev/urandom", "r");
    if (!f) {
        perror("fopen /dev/urandom");
        return 1;
    }

    ret = fseek(f, /*offset=*/0, SEEK_CUR);
    if (ret < 0) {
        perror("fseek /dev/urandom");
        return 1;
    }

    memset(buf, 0, sizeof(buf));
    n = fread(buf, 1, sizeof(buf), f);
    if (n < sizeof(buf) || ferror(f)) {
        perror("fread /dev/urandom");
        return 1;
    }

    printf("Four bytes from /dev/urandom:");
    for (int i = 0; i < 4; i++)
        printf(" %02hhx", (unsigned char)buf[i]);
    printf("\n");

    ret = fclose(f);
    if (ret) {
        perror("fclose /dev/urandom");
        return 1;
    }

    printf("===== Write to /dev/null\n");
    f = fopen("/dev/null", "w");
    if (!f) {
        perror("fopen /dev/null");
        return 1;
    }

    ret = fseek(f, /*offset=*/0, SEEK_CUR);
    if (ret < 0) {
        perror("fseek /dev/null");
        return 1;
    }

    n = fwrite(buf, 1, sizeof(buf), f);
    if (n < sizeof(buf)) {
        perror("fwrite /dev/null");
        return 1;
    }

    ret = fclose(f);
    if (ret) {
        perror("fclose /dev/null");
        return 1;
    }

#if 0
    /* `/dev/stdout` is a link to `/proc/self/fd/1`; Gramine currently fails, see #1387 */

    printf("===== Write to /dev/stdout\n");
    f = fopen("/dev/stdout", "w");
    if (!f) {
        perror("fopen /dev/stdout");
        return 1;
    }

    sprintf(buf, "%s\n", "Hello World written to stdout!");
    size_t towrite = strlen(buf) + 1;
    n = fwrite(buf, 1, towrite, f);
    if (n < towrite) {
        perror("fwrite /dev/stdout");
        return 1;
    }

    ret = fclose(f);
    if (ret) {
        perror("fclose /dev/stdout");
        return 1;
    }
#endif

    puts("TEST OK!");
    return 0;
}
