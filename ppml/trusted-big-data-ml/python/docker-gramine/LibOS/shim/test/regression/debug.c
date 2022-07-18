#include <stdio.h>

__attribute__((noinline)) static void func(void) {
    printf("hello\n");
    fflush(stdout);
    /* Prevent tail call, so that we get a stack trace including func(). */
    __asm__ volatile("");
}

int main(void) {
    func();
    return 0;
}
