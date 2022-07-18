#include <iostream>
#include <stdexcept>

extern "C" void f2() {
    throw std::runtime_error("test runtime error");
}

extern "C" void f1();
/*
 * This function is written in asm to make sure that it saves a register on stack and has proper
 * cfi. The reason is to force stack unwinding routines to touch memory (to get the previous
 * register value), which can have unexpected side effects (e.g. libunwind uses `msync` syscall
 * to check if the memory address is valid).
 */
__asm__ (
".pushsection .text\n"
".global f1\n"
".type f1, @function\n"
"f1:\n"
".cfi_startproc\n"
"push %rbp\n"
".cfi_adjust_cfa_offset 8\n"
".cfi_rel_offset %rbp, 0\n"

"mov $0x1337, %rbp\n"
"call f2\n"

"pop %rbp\n"
".cfi_adjust_cfa_offset -8\n"
".cfi_restore %rbp\n"
"ret\n"
".cfi_endproc\n"
".popsection\n"
);

int main() {
    std::cout << "User Program Started" << std::endl;

    try {
        f1();
    } catch (const std::exception& e) {
        std::cout << "Exception '" << e.what() << "' caught" << std::endl;
    }

    return 0;
}
