#include "api.h"
#include "pal.h"
#include "pal_regression.h"

char buffer[80];

int main(int argc, char** argv, char** envp) {
    /* test regular directory opening */

    PAL_HANDLE dir1 = NULL;
    int ret = DkStreamOpen("dir:dir_exist.tmp", PAL_ACCESS_RDONLY, /*share_flags=*/0,
                           PAL_CREATE_NEVER, /*options=*/0, &dir1);
    if (ret >= 0 && dir1) {
        pal_printf("Directory Open Test 1 OK\n");

        PAL_STREAM_ATTR attr1;
        ret = DkStreamAttributesQueryByHandle(dir1, &attr1);
        if (ret >= 0) {
            pal_printf("Query by Handle: type = %d\n", attr1.handle_type);
        }

        size_t bytes = sizeof(buffer);
        ret = DkStreamRead(dir1, 0, &bytes, buffer, NULL, 0);
        if (ret >= 0 && bytes) {
            for (char* c = buffer; c < buffer + bytes; c += strlen(c) + 1)
                if (strlen(c))
                    pal_printf("Read Directory: %s\n", c);
        }

        DkObjectClose(dir1);
    }

    PAL_HANDLE dir2 = NULL;
    ret = DkStreamOpen("dir:./dir_exist.tmp", PAL_ACCESS_RDONLY, /*share_flags=*/0,
                       PAL_CREATE_NEVER, /*options=*/0, &dir2);
    if (ret >= 0 && dir2) {
        pal_printf("Directory Open Test 2 OK\n");
        DkObjectClose(dir2);
    }

    PAL_HANDLE dir3 = NULL;
    ret = DkStreamOpen("dir:../regression/dir_exist.tmp", PAL_ACCESS_RDONLY, /*share_flags=*/0,
                       PAL_CREATE_NEVER, /*options=*/0, &dir3);
    if (ret >= 0 && dir3) {
        pal_printf("Directory Open Test 3 OK\n");
        DkObjectClose(dir3);
    }

    PAL_STREAM_ATTR attr2;
    ret = DkStreamAttributesQuery("dir:dir_exist.tmp", &attr2);
    if (ret >= 0) {
        pal_printf("Query: type = %d\n", attr2.handle_type);
    }

    /* test regular directory creation */

    PAL_HANDLE dir4 = NULL;
    ret = DkStreamOpen("dir:dir_nonexist.tmp", PAL_ACCESS_RDONLY,
                       PAL_SHARE_OWNER_R | PAL_SHARE_OWNER_W | PAL_SHARE_OWNER_X,
                       PAL_CREATE_ALWAYS, /*options=*/0, &dir4);
    if (ret >= 0 && dir4) {
        pal_printf("Directory Creation Test 1 OK\n");
        DkObjectClose(dir4);
    }

    PAL_HANDLE dir5 = NULL;
    ret = DkStreamOpen("dir:dir_nonexist.tmp", PAL_ACCESS_RDONLY,
                       PAL_SHARE_OWNER_R | PAL_SHARE_OWNER_W | PAL_SHARE_OWNER_X,
                       PAL_CREATE_ALWAYS, /*options=*/0, &dir5);
    if (ret >= 0) {
        DkObjectClose(dir5);
    } else {
        pal_printf("Directory Creation Test 2 OK\n");
    }

    PAL_HANDLE dir6 = NULL;
    ret = DkStreamOpen("dir:dir_nonexist.tmp", PAL_ACCESS_RDWR,
                       PAL_SHARE_OWNER_R | PAL_SHARE_OWNER_W, PAL_CREATE_TRY, /*options=*/0, &dir6);
    if (ret >= 0 && dir6) {
        pal_printf("Directory Creation Test 3 OK\n");
        DkObjectClose(dir6);
    }

    PAL_HANDLE dir7 = NULL;
    ret = DkStreamOpen("dir:dir_delete.tmp", PAL_ACCESS_RDONLY, /*share_flags=*/0,
                       PAL_CREATE_NEVER, /*options=*/0, &dir7);
    if (ret >= 0 && dir7) {
        ret = DkStreamDelete(dir7, PAL_DELETE_ALL);
        if (ret < 0) {
            pal_printf("DkStreamDelete failed\n");
            return 1;
        }
        DkObjectClose(dir7);
    }

    return 0;
}
