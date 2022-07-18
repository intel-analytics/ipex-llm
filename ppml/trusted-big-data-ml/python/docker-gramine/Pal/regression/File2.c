#include "pal.h"
#include "pal_regression.h"

#define FILE_URI "file:test.txt"

char str[12] = "Hello World";

int main(int argc, char** argv, char** envp) {
    pal_printf("Enter Main Thread\n");

    PAL_HANDLE out = NULL;
    int ret = DkStreamOpen(FILE_URI, PAL_ACCESS_RDWR, PAL_SHARE_OWNER_W | PAL_SHARE_OWNER_R,
                           PAL_CREATE_TRY, /*options=*/0, &out);

    if (ret < 0) {
        pal_printf("first DkStreamOpen failed\n");
        return 1;
    }

    size_t bytes = sizeof(str) - 1;
    ret = DkStreamWrite(out, 0, &bytes, str, NULL);
    if (ret < 0 || bytes != sizeof(str) - 1) {
        pal_printf("second DkStreamWrite failed\n");
        return 1;
    }

    DkObjectClose(out);

    PAL_HANDLE in = NULL;
    ret = DkStreamOpen(FILE_URI, PAL_ACCESS_RDONLY, /*share_flags=*/0, PAL_CREATE_NEVER,
                       /*options=*/0, &in);
    if (ret < 0) {
        pal_printf("third DkStreamOpen failed\n");
        return 1;
    }

    bytes = sizeof(str);
    memset(str, 0, bytes);
    ret = DkStreamRead(in, 0, &bytes, str, NULL, 0);
    if (ret < 0) {
        pal_printf("DkStreamRead failed\n");
        return 1;
    }
    if (bytes > sizeof(str) - 1) {
        pal_printf("DkStreamRead read more than expected\n");
        return 1;
    }
    str[bytes] = '\0';

    pal_printf("%s\n", str);

    ret = DkStreamDelete(in, PAL_DELETE_ALL);
    if (ret < 0) {
        pal_printf("DkStreamDelete failed\n");
        return 1;
    }

    PAL_HANDLE del = NULL;
    ret = DkStreamOpen(FILE_URI, PAL_ACCESS_RDWR, /*share_flags=*/0, PAL_CREATE_NEVER,
                       /*options=*/0, &del);

    if (ret >= 0) {
        pal_printf("DkStreamDelete did not actually delete\n");
        return 1;
    }

    pal_printf("Leave Main Thread\n");
    return 0;
}
