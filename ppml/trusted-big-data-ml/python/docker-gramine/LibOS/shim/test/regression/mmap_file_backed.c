#define _GNU_SOURCE
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char** argv) {
    int ret;

    if (argc != 2)
        errx(1, "Usage: %s file_to_mmap", argv[0]);

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size < 0)
        err(1, "sysconf");

    /* open the file for read, find its size, mmap at least one page more than the file size and
     * mprotect the last page */
    FILE* fp = fopen(argv[1], "r");
    if (!fp)
        err(1, "fopen");

    ret = fseek(fp, 0, SEEK_END);
    if (ret < 0)
        err(1, "fseek");

    long fsize = ftell(fp);
    if (fsize < 0)
        err(1, "ftell");

    rewind(fp); /* for sanity */

    size_t mmap_size = fsize + page_size * 2; /* file size plus at least one full aligned page */
    mmap_size &= ~(page_size - 1);            /* align down */

    /* mmap with write-only protections: we want to taint the mmapped region; we cannot use
     * read-write permissions because this is disallowed for SGX protected files */
    void* addr = mmap(NULL, mmap_size, PROT_WRITE, MAP_PRIVATE, fileno(fp), 0);
    if (addr == MAP_FAILED)
        err(1, "mmap");

    ret = mprotect(addr + mmap_size - page_size, page_size, PROT_NONE);
    if (ret < 0)
        err(1, "mprotect");

    /* Below fork triggers checkpoint-and-restore logic in Gramine LibOS, which will send all VMAs
     * info and all corresponding memory contents to the child. These VMAs contain two VMAs that
     * were split from the file-backed mmap above: the first VMA with the lower part (backed by file
     * contents, except one or two last pages that may be not backed completely) and the second VMA
     * with a single page (not backed by file contents).
     *
     * There was a bug in Gramine-SGX: mmap(..., <file-fd>, <offset-past-file-end>) during
     * checkpoint restore in the child crashed on trusted files. So the below fork checks that this
     * bug was fixed. */
    int pid = fork();
    if (pid == -1)
        err(1, "fork");

    if (pid != 0) {
        /* parent */
        int st = 0;
        ret = wait(&st);
        if (ret < 0)
            err(1, "wait");

        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
            errx(1, "abnormal child termination: %d", st);

        puts("Parent process done");
    } else {
        /* child does nothing interesting */
        puts("Child process done");
    }

    fclose(fp);
    return 0;
}
