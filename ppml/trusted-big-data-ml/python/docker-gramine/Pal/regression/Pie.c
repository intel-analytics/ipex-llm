#include "pal.h"
#include "pal_regression.h"

char str[13] = "Hello World\n";

int main(int argc, char** argv, char** envp) {
    pal_printf("start program: Pie\n");

    PAL_HANDLE out = NULL;
    int ret = DkStreamOpen("dev:tty", PAL_ACCESS_WRONLY, /*share_flags=*/0, PAL_CREATE_NEVER,
                           /*options=*/0, &out);

    if (ret < 0) {
        pal_printf("DkStreamOpen failed\n");
        return 1;
    }

    size_t bytes = sizeof(str) - 1;
    ret = DkStreamWrite(out, 0, &bytes, str, NULL);

    if (ret < 0 || bytes != sizeof(str) - 1) {
        pal_printf("DkStreamWrite failed\n");
        return 1;
    }

    DkObjectClose(out);
    return 0;
}
