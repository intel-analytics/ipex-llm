#include "pal.h"
#include "pal_regression.h"

#define SYMBOL_ADDR(sym)                                                    \
    ({                                                                      \
        void* _sym;                                                         \
        __asm__ volatile("movq " #sym "@GOTPCREL(%%rip), %0" : "=r"(_sym)); \
        _sym;                                                               \
    })

#define PRINT_SYMBOL(sym) pal_printf("symbol: %s = %p\n", #sym, SYMBOL_ADDR(sym))

int main(int argc, char** argv, char** envp) {
    PRINT_SYMBOL(DkVirtualMemoryAlloc);
    PRINT_SYMBOL(DkVirtualMemoryFree);
    PRINT_SYMBOL(DkVirtualMemoryProtect);

    PRINT_SYMBOL(DkProcessCreate);
    PRINT_SYMBOL(DkProcessExit);

    PRINT_SYMBOL(DkStreamOpen);
    PRINT_SYMBOL(DkStreamWaitForClient);
    PRINT_SYMBOL(DkStreamRead);
    PRINT_SYMBOL(DkStreamWrite);
    PRINT_SYMBOL(DkStreamDelete);
    PRINT_SYMBOL(DkStreamMap);
    PRINT_SYMBOL(DkStreamUnmap);
    PRINT_SYMBOL(DkStreamSetLength);
    PRINT_SYMBOL(DkStreamFlush);
    PRINT_SYMBOL(DkSendHandle);
    PRINT_SYMBOL(DkReceiveHandle);
    PRINT_SYMBOL(DkStreamAttributesQuery);
    PRINT_SYMBOL(DkStreamAttributesQueryByHandle);
    PRINT_SYMBOL(DkStreamAttributesSetByHandle);
    PRINT_SYMBOL(DkStreamGetName);
    PRINT_SYMBOL(DkStreamChangeName);
    PRINT_SYMBOL(DkStreamsWaitEvents);

    PRINT_SYMBOL(DkThreadCreate);
    PRINT_SYMBOL(DkThreadYieldExecution);
    PRINT_SYMBOL(DkThreadExit);
    PRINT_SYMBOL(DkThreadResume);

    PRINT_SYMBOL(DkSetExceptionHandler);

    PRINT_SYMBOL(DkEventCreate);
    PRINT_SYMBOL(DkEventSet);
    PRINT_SYMBOL(DkEventClear);
    PRINT_SYMBOL(DkEventWait);

    PRINT_SYMBOL(DkObjectClose);

    PRINT_SYMBOL(DkSystemTimeQuery);
    PRINT_SYMBOL(DkRandomBitsRead);
#if defined(__x86_64__)
    PRINT_SYMBOL(DkSegmentBaseGet);
    PRINT_SYMBOL(DkSegmentBaseSet);
#endif
    PRINT_SYMBOL(DkMemoryAvailableQuota);

    return 0;
}
