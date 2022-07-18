#include <stdlib.h>

/* Test if the correct error code is propagated on a signal-induced exit. This should report 134
 * (128 + 6 where 6 is SIGABRT) as its return code. */

int main(int argc, char* arvg[]) {
    abort();
}
