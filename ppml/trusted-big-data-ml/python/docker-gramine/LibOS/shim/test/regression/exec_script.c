#include <err.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    char* const argv[] = {(char*)"scripts/foo.sh", (char*)"STRING FROM EXECVE", NULL};
    execv(argv[0], argv);
    err(EXIT_FAILURE, "execve failed");
}
