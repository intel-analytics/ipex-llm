#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char** argv) {
    struct stat root, proc_root;

    if (stat("/", &root) == -1) {
        perror("stat");
        exit(1);
    }

    if (stat("/proc/1/root", &proc_root) == -1) {
        perror("stat");
        exit(1);
    }

    if (root.st_dev == proc_root.st_dev && root.st_ino == proc_root.st_ino) {
        printf("proc path test success\n");
    } else {
        printf("proc path test failure\n");
    }
    return 0;
}
