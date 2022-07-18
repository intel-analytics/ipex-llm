#include "api.h"
#include "pal_regression.h"

#define FAIL(code, fmt...) ({   \
    pal_printf(fmt);            \
    pal_printf("\n");           \
    DkProcessExit(code);        \
})

#define TEST(output_str, fmt...) ({                                                                \
    size_t output_len = strlen(output_str);                                                        \
    char buf[0x100];                                                                               \
    int x = snprintf(buf, sizeof(buf) - 1,  fmt);                                                  \
    buf[sizeof(buf) - 1] = 0;                                                                      \
    if (x < 0 || (size_t)x != output_len) {                                                        \
        FAIL(1, "wrong return val at %d, expected %zu, got %d", __LINE__, output_len, x);          \
    }                                                                                              \
    if (strcmp(buf, output_str)) {                                                                 \
        FAIL(1, "wrong output string at %d, expected \"%s\", got \"%s\"", __LINE__, output_str,    \
                buf);                                                                              \
    }                                                                                              \
})

int main(void) {
    char* ptr = NULL;

    /* Basic tests. */
    TEST("1337", "%u", 1337);
    TEST("1337", "%d", 1337);
    TEST("1337", "%i", 1337);
    TEST("-1337", "%d", -1337);
    TEST("1337", "%x", 0x1337);
    TEST("1337", "%o", 01337);
    TEST("ab\n", "%c%c%c", 0x61, 0x62, 10);
    TEST("asdf", "%s", "asdf");
    TEST("0x1337", "%p", (void*)0x1337);
    TEST("ab%cd", "ab%%cd");
    TEST("(null)", "%s", ptr);

    /* Length modifiers. */
    TEST("-13", "%hhd", (signed char)-13);
    TEST("13", "%hu", (unsigned short)13);
    TEST("1337", "%lx", 0x1337ul);
    TEST("18446744073709551615", "%llu", 18446744073709551615ull);
    TEST("18446744073709551615", "%zu", (size_t)18446744073709551615ull);

    /* Width & precision. */
    TEST("    1337", "%8u", 1337);
    TEST("13371337", "%3u", 13371337);
    TEST("13371337", "%0u", 13371337);
    TEST("000001337", "%.9u", 1337);
    TEST("1337", "%.0d", 1337);
    TEST("0", "%.1d", 0);
    TEST("", "%.0d", 0);
    TEST("", "%.0x", 0);
    TEST("      abcde", "%11s", "abcde");
    TEST("abc", "%.3s", "abcde");
    TEST(" -13", "%*d", 4, -13);
    TEST("-0013", "%.*d", 4, -13);
    TEST("   abc", "%*.*s", 6, 3, "abcde");

    /* Flags. */
    TEST("0x1337", "%#x", 0x1337);
    TEST("01337", "%#o", 01337);
    TEST("0", "%#x", 0);
    TEST("", "%#.0x", 0);
    TEST("0", "%#o", 0);
    TEST("0", "%#.0o", 0);
    TEST("00", "%#02x", 0);
    TEST("000", "%#.3o", 0);
    TEST("00001337", "%08u", 1337);
    TEST("+1337", "%+d", 1337);
    TEST("-1337", "%+d", -1337);
    TEST(" 1337", "% d", 1337);
    TEST("-1337", "% d", -1337);
    TEST("1337    |", "%-8u|", 1337);

    /* Random mixture of everything. */
    TEST(" 0x1337", "%#7x", 0x1337);
    TEST("0x0001337", "%#.7x", 0x1337);
    TEST(" 0001337  |", "% -10.7ld|", 1337l);
    TEST(" 0001337  |", "% -*.*ld|", 10, 7, 1337l);
    TEST("ab   0x13cd", "ab%7pcd", (void*)0x13);
    TEST("  0x001337", "%#10.6x", 0x1337);
    TEST("0x001337  ", "%#-10.6x", 0x1337);

    /* This one differs from the "normal" printf, but as far as I can tell, it matches the C
     * standard. */
    TEST("000x1337", "%#08x", 0x1337);

    /* Test whether string precision is correctly respected. Create a string that is not null
     * terminated at two pages boundary, where the second page has no read permission. If
     * the precision limit is not respected, `snprintf` will access data at the second page and
     * crash the process. */
    int ret = DkVirtualMemoryAlloc((void**)&ptr, 2 * PAGE_SIZE, PAL_ALLOC_INTERNAL,
                                   PAL_PROT_READ | PAL_PROT_WRITE);
    if (ret < 0) {
        FAIL(1, "DkVirtualMemoryAlloc failed: %d", ret);
    }
    ret = DkVirtualMemoryProtect(ptr + PAGE_SIZE, PAGE_SIZE, /*prot=*/0);
    if (ret < 0) {
        FAIL(1, "DkVirtualMemoryProtect failed: %d", ret);
    }
    memset(ptr + PAGE_SIZE - 7, 'a', 7);

    ret = snprintf(ptr, PAGE_SIZE - 8, "%.7s", ptr + PAGE_SIZE - 7);
    if (ret != 7) {
        FAIL(1, "snprintf at %d returned %d, expected %d", __LINE__, ret, 7);
    }

    pal_printf("TEST OK\n");
    return 0;
}
