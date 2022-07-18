#include <stdio.h>

static void* ptr = NULL;

int main(int argc, const char** argv, const char** envp) {
    printf("User program started\n");
    printf("Local Address in Executable: %p\n", &ptr);

    for (int i = 0; i < argc; i++) {
        printf("argv[%d] = %s\n", i, argv[i]);
    }

    return 0;
}
