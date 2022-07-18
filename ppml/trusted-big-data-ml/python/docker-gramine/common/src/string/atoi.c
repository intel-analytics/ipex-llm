/* Copyright (C) 2021 Intel Corporation
 *                    Vijay Dhanraj <vijay.dhanraj@intel.com>
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#include <limits.h>
#include <stdint.h>

#include "api.h"

static void begin_number(const char* str, int base, const char** out_s, int* out_base,
                         int* out_sign) {
    const char* s = str;

    // gobble initial whitespace
    while (*s == ' ' || *s == '\t') {
        s++;
    }

    // plus/minus sign
    int sign = 1;
    if (*s == '+') {
        s++;
    } else if (*s == '-') {
        s++;
        sign = -1;
    }

    // hex or octal base prefix
    if ((base == 0 || base == 16) && (s[0] == '0' && s[1] == 'x')) {
        s += 2;
        base = 16;
    } else if (base == 0 && s[0] == '0') {
        s++;
        base = 8;
    } else if (base == 0) {
        base = 10;
    }

    *out_s = s;
    *out_base = base;
    *out_sign = sign;
}

static int parse_digit(char c, int base) {
    int digit;

    if (c >= '0' && c <= '9') {
        digit = c - '0';
    } else if (c >= 'a' && c <= 'z') {
        digit = c - 'a' + 10;
    } else if (c >= 'A' && c <= 'Z') {
        digit = c - 'A' + 10;
    } else {
        return -1;
    }
    if (digit >= base)
        return -1;
    return digit;
}

long strtol(const char* str, char** out_end, int base) {
    const char* s;
    int sign;

    begin_number(str, base, &s, &base, &sign);

    long value = 0;
    while (*s != '\0') {
        int digit = parse_digit(*s, base);
        if (digit == -1)
            break;

        if (__builtin_mul_overflow(value, base, &value)) {
            return sign > 0 ? LONG_MAX : LONG_MIN;
        }

        if (__builtin_add_overflow(value, digit * sign, &value)) {
            return sign > 0 ? LONG_MAX : LONG_MIN;
        }

        s++;
    }

    if (out_end)
        *out_end = (char*)s;
    return value;
}

int str_to_ulong(const char* str, unsigned int base, unsigned long* out_value,
                 const char** out_end) {
    if (base == 16 && str[0] == '0' && str[1] == 'x')
        str += 2;

    unsigned long value = 0;
    const char* s = str;
    while (*s != '\0') {
        int digit = parse_digit(*s, base);
        if (digit == -1)
            break;

        if (__builtin_mul_overflow(value, base, &value))
            return -1;

        if (__builtin_add_overflow(value, digit, &value))
            return -1;

        s++;
    }

    if (s == str)
        return -1;

    *out_value = value;
    *out_end = s;
    return 0;
}

#ifdef __LP64__
/* long int == long long int on targets with data model LP64 */
long long strtoll(const char* s, char** endptr, int base) {
    return (long long)strtol(s, endptr, base);
}
#else
#error "Unsupported architecture (only support data model LP64)"
#endif

/* Convert a string to an int (without error checking). */
int atoi(const char* str) {
    return (int)atol(str);
}

/* Convert a string to a long int (without error checking). */
long int atol(const char* str) {
    const char* s;
    int sign;
    int base;
    begin_number(str, 10, &s, &base, &sign);
    assert(base == 10);

    long value = 0;
    while (*s != '\0') {
        int digit = parse_digit(*s, 10);
        if (digit == -1)
            break;

        value *= 10;
        value += digit * sign;

        s++;
    }
    return value;
}

int parse_size_str(const char* str, uint64_t* out_val) {
    const char* endptr = NULL;
    unsigned long size;
    int ret = str_to_ulong(str, 10, &size, &endptr);
    if (ret < 0)
        return -1;

    unsigned long unit = 1;
    if (*endptr == 'G' || *endptr == 'g') {
        unit = 1024 * 1024 * 1024;
        endptr++;
    } else if (*endptr == 'M' || *endptr == 'm') {
        unit = 1024 * 1024;
        endptr++;
    } else if (*endptr == 'K' || *endptr == 'k') {
        unit = 1024;
        endptr++;
    }

    if (__builtin_mul_overflow(size, unit, &size))
        return -1;

    if (*endptr != '\0')
        return -1; /* garbage found after the size string */

    if (OVERFLOWS(__typeof__(*out_val), size))
        return -1;

    *out_val = size;
    return 0;
}
