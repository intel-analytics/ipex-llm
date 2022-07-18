#include <assert.h>

#include "hex.h"
#include "pal.h"
#include "pal_regression.h"

char x[] = {0xde, 0xad, 0xbe, 0xef};
char y[] = {0xcd, 0xcd, 0xcd, 0xcd, 0xcd, 0xcd, 0xcd, 0xcd};

static_assert(sizeof(x) <= sizeof(y), "array x is longer than array y");
char hex_buf[sizeof(y) * 2 + 1];

int main(void) {
    pal_printf("Hex test 1 is %s\n", BYTES2HEXSTR(x, hex_buf, sizeof(hex_buf)));
    pal_printf("Hex test 2 is %s\n", BYTES2HEXSTR(y, hex_buf, sizeof(hex_buf)));
    return 0;
}
