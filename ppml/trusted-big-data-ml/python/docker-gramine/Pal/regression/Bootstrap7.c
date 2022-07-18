#include "pal.h"
#include "pal_regression.h"

int main(int argc, char** argv, char** envp) {
    pal_printf("User Program Started\n");

    for (int i = 0; envp[i]; i++)
        pal_printf("%s\n", envp[i]);

    return 0;
}
