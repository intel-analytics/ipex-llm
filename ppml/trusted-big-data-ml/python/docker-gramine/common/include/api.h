/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

#ifndef API_H
#define API_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef USE_STDLIB
#include <assert.h>
#else
#include "assert.h"
#endif

#include "cpu.h"
#include "list.h"
#include "log.h"

/* WARNING: this declaration may conflict with some header files */
#ifndef ssize_t
typedef ptrdiff_t ssize_t;
#endif

/* Macros */

#ifndef MIN
#define MIN(a, b)               \
    ({                          \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a < _b ? _a : _b;      \
    })
#endif
#ifndef MAX
#define MAX(a, b)               \
    ({                          \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a > _b ? _a : _b;      \
    })
#endif

#define SATURATED_ADD(a, b, limit)                                    \
    ({                                                                \
        __typeof__(a) _a = (a);                                       \
        __typeof__(b) _b = (b);                                       \
        __typeof__(limit) _limit = (limit);                           \
        _b > _limit ? _limit : (_a > _limit - _b ? _limit : _a + _b); \
    })

#define SATURATED_SUB(a, b, limit)                                    \
    ({                                                                \
        __typeof__(a) _a = (a);                                       \
        __typeof__(b) _b = (b);                                       \
        __typeof__(limit) _limit = (limit);                           \
        _a < _limit ? _limit : (_b > _a - _limit ? _limit : _a - _b); \
    })

#define SATURATED_P_ADD(ptr_a, b, limit) \
    ((__typeof__(ptr_a))SATURATED_ADD((uintptr_t)(ptr_a), (uintptr_t)(b), (uintptr_t)(limit)))

#define SATURATED_P_SUB(ptr_a, b, limit) \
    ((__typeof__(ptr_a))SATURATED_SUB((uintptr_t)(ptr_a), (uintptr_t)(b), (uintptr_t)(limit)))

#define OVERFLOWS(type, val)                        \
    ({                                              \
        type __dummy;                               \
        __builtin_add_overflow((val), 0, &__dummy); \
    })

#define IS_POWER_OF_2(x)          \
    ({                            \
        assert((x) != 0);         \
        (((x) & ((x) - 1)) == 0); \
    })

/* Safe against overflows. May be slower than `(((n) + (d) - 1) / (d))` (which is not
 * overflow-safe!), but hopefully the compiler will merge division and modulo into one
 * instruction. */
#define UDIV_ROUND_UP(n, d)             ((n) / (d) + !!((n) % (d)))

#define BITS_IN_BYTE                    8
#define BITS_IN_TYPE(type)              (sizeof(type) * BITS_IN_BYTE)
#define BITS_TO_UINT32S(nr)             UDIV_ROUND_UP(nr, BITS_IN_TYPE(uint32_t))
#define BITS_TO_LONGS(nr)               UDIV_ROUND_UP(nr, BITS_IN_TYPE(long))
/* Note: This macro is not intended for use when nbits == BITS_IN_TYPE(type) */
#define SET_HIGHEST_N_BITS(type, nbits) (~(((uint64_t)1 << (BITS_IN_TYPE(type) - (nbits))) - 1))
#define WITHIN_MASK(val, mask) (((val) | (mask)) == (mask))

#define IS_ALIGNED(val, alignment)     ((val) % (alignment) == 0)
#define ALIGN_DOWN(val, alignment)     ((val) - (val) % (alignment))
#define ALIGN_UP(val, alignment)       ALIGN_DOWN((val) + (alignment) - 1, alignment)
#define IS_ALIGNED_PTR(val, alignment) IS_ALIGNED((uintptr_t)(val), alignment)
#define ALIGN_DOWN_PTR(ptr, alignment) ((__typeof__(ptr))(ALIGN_DOWN((uintptr_t)(ptr), alignment)))
#define ALIGN_UP_PTR(ptr, alignment)   ((__typeof__(ptr))(ALIGN_UP((uintptr_t)(ptr), alignment)))

/* Useful only when the alignment is a power of two, but when that's not known compile-time. */
#define IS_ALIGNED_POW2(val, alignment) (((val) & ((alignment) - 1)) == 0)
#define ALIGN_DOWN_POW2(val, alignment) \
    ((val) - ((val) & ((alignment) - 1))) // `~` doesn't work if `alignment` is of a smaller type
                                          // than `val` and unsigned.
#define ALIGN_UP_POW2(val, alignment)       ALIGN_DOWN_POW2((val) + (alignment) - 1, alignment)
#define IS_ALIGNED_PTR_POW2(val, alignment) IS_ALIGNED_POW2((uintptr_t)(val), alignment)
#define ALIGN_DOWN_PTR_POW2(ptr, alignment) \
    ((__typeof__(ptr))(ALIGN_DOWN_POW2((uintptr_t)(ptr), alignment)))
#define ALIGN_UP_PTR_POW2(ptr, alignment) \
    ((__typeof__(ptr))(ALIGN_UP_POW2((uintptr_t)(ptr), alignment)))

#define SAME_TYPE(a, b)       __builtin_types_compatible_p(__typeof__(a), __typeof__(b))
#define IS_STATIC_ARRAY(a)    (!SAME_TYPE(a, &*(a)))
#define FORCE_STATIC_ARRAY(a) sizeof(int[IS_STATIC_ARRAY(a) - 1]) // evaluates to 0

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(a) (FORCE_STATIC_ARRAY(a) + sizeof(a) / sizeof(a[0]))
#endif

#define IS_SIGNED(T) ((T)-1 < (T)1)

#define SET_UNALIGNED(a, b) ({                  \
    __typeof__(b) _b = (b);                     \
    static_assert(SAME_TYPE((a), _b), "error"); \
    memcpy(&(a), &_b, sizeof(a));               \
})

#define GET_UNALIGNED(a) ({             \
    __typeof__(a) ret;                  \
    memcpy(&ret, &(a), sizeof(ret));    \
    ret;                                \
})

#define DEBUG_BREAK()               \
    do {                            \
        __asm__ volatile("int $3"); \
    } while (0)

#if 0
#define DEBUG_BREAK_ON_FAILURE() DEBUG_BREAK()
#else
#define DEBUG_BREAK_ON_FAILURE() do {} while (0)
#endif

#define BUG()                                           \
    do {                                                \
        log_error("BUG() %s:%d", __FILE__, __LINE__);   \
        DEBUG_BREAK_ON_FAILURE();                       \
        die_or_inf_loop();                              \
    } while (0)

#ifndef container_of
/**
 * container_of - cast a member of a structure out to the containing structure
 * @ptr:    the pointer to the member.
 * @type:   the type of the container struct this is embedded in.
 * @member: the name of the member within the struct.
 *
 */
#define container_of(ptr, type, member) ((type*)((char*)(ptr) - offsetof(type, member)))
#endif

#define __alloca __builtin_alloca

/* Clang has different syntax than GCC for no-stack-protector, see:
 * https://reviews.llvm.org/D46300 */
#ifdef __clang__
#define __attribute_no_stack_protector __attribute((no_stack_protector))
#else
#define __attribute_no_stack_protector __attribute__((__optimize__("-fno-stack-protector")))
#endif

#ifdef __clang__
#define __attribute_no_sanitize_address __attribute((no_sanitize("address")))
#else
/* We support ASan only for Clang (see `asan.h`), and older GCC versions actually do not know this
 * attribute. */
#define __attribute_no_sanitize_address
#endif

#define XSTRINGIFY(x) STRINGIFY(x)
#define STRINGIFY(x)  #x

/* fail build if str is not a static string */
#define FORCE_LITERAL_CSTR(str) ("" str "")

#define __UNUSED(x) \
    do {            \
        (void)(x);  \
    } while (0)
#define static_strlen(str) (ARRAY_SIZE(FORCE_LITERAL_CSTR(str)) - 1)

#define IS_IN_RANGE_INCL(value, start, end) (((value) < (start) || (value) > (end)) ? false : true)

/* Each occurence of this macro in the source code will return `true` only once per process.
 *
 * We use __ATOMIC_RELAXED here, as a consistent ordering within the accesses to `first` is enough
 * for us — FIRST_TIME is not a synchronization primitive, as the macro returning `false` doesn't
 * actually guarantee that the code path entered when it returned `true` has finished executing. */
#define FIRST_TIME() ({ static uint8_t first = 0; __atomic_exchange_n(&first, 1, __ATOMIC_RELAXED) == 0; })

/* LibC functions */

/* LibC string functions */
size_t strnlen(const char* str, size_t maxlen);
size_t strlen(const char* str);
int strncmp(const char* lhs, const char* rhs, size_t maxlen);
int strcmp(const char* lhs, const char* rhs);

long strtol(const char* s, char** endptr, int base);
long long strtoll(const char* s, char** endptr, int base);

/*!
 * \brief Convert a string to number.
 *
 * \param      str        Input string.
 * \param      base       Digit base, between 2 and 36.
 * \param[out] out_value  On success, set to the parsed number.
 * \param[out] out_end    On success, set to the rest of string.
 *
 * \returns 0 on success, negative on failure.
 *
 * Parses a number from the beginning of a string. The number should be non-empty, consist of digits
 * only (no `+`/`-` signs), and not overflow the `unsigned long` type. For base 16, the "0x" prefix
 * is allowed but not required.
 */
int str_to_ulong(const char* str, unsigned int base, unsigned long* out_value,
                 const char** out_end);

int atoi(const char* nptr);
long int atol(const char* nptr);

int islower(int c);
int isupper(int c);
int tolower(int c);
int toupper(int c);
int isalpha(int c);
int isdigit(int c);
int isalnum(int c);

char* strchr(const char* s, int c);
char* strstr(const char* haystack, const char* needle);
size_t strspn(const char* s, const char* c);

void* memcpy(void* restrict dest, const void* restrict src, size_t count);
void* memmove(void* dest, const void* src, size_t count);
void* memset(void* dest, int ch, size_t count);
int memcmp(const void* lhs, const void* rhs, size_t count);

/* Used by _FORTIFY_SOURCE */
void* __memcpy_chk(void* restrict dest, const void* restrict src, size_t count, size_t dest_count);
void* __memmove_chk(void* dest, const void* src, size_t count, size_t dest_count);
void* __memset_chk(void* dest, int ch, size_t count, size_t dest_count);

/* Original versions of functions that ASan overrides */
void* _real_memcpy(void* restrict dest, const void* restrict src, size_t count);
void* _real_memmove(void* dest, const void* src, size_t count);
void* _real_memset(void* dest, int ch, size_t count);
int _real_memcmp(const void* lhs, const void* rhs, size_t count);

bool strstartswith(const char* str, const char* prefix);
bool strendswith(const char* str, const char* suffix);
char* strdup(const char* str);
char* alloc_substr(const char* start, size_t len);
char* alloc_concat(const char* a, size_t a_len, const char* b, size_t b_len);
char* alloc_concat3(const char* a, size_t a_len, const char* b, size_t b_len,
                    const char* c, size_t c_len);

/* Libc memory allocation functions */
void* malloc(size_t size);
void free(void* ptr);
void* calloc(size_t nmemb, size_t size);

/* copy static string and return the address of the NUL byte (NULL if the dest
 * is not large enough).*/
#define strcpy_static(var, str, max)                                  \
    (static_strlen(str) + 1 > (max)                                   \
     ? NULL                                                           \
     : memcpy(var, str, static_strlen(str) + 1) + static_strlen(str))

/* Copy a fixed size array. */
#define COPY_ARRAY(dst, src)                                                   \
    do {                                                                       \
        /* Using pointers because otherwise the compiler would try to allocate \
         * memory for the fixed size arrays and complain about invalid         \
         * initializers.                                                       \
         */                                                                    \
        __typeof__(src)* _s = &(src);                                          \
        __typeof__(dst)* _d = &(dst);                                          \
                                                                               \
        static_assert(SAME_TYPE((*_s)[0], (*_d)[0]), "types must match");      \
        static_assert(ARRAY_SIZE(*_s) == ARRAY_SIZE(*_d), "sizes must match"); \
                                                                               \
        memcpy(*_d, *_s, sizeof(*_d));                                         \
    } while (0)

#define COMPILER_BARRIER() ({ __asm__ __volatile__("" ::: "memory"); })

/* We need this artificial assignment in READ_ONCE because of a GCC bug:
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=99258
 */
#define READ_ONCE(x) ({ __typeof__(x) y = *(volatile __typeof__(x)*)&(x); y;})

#define WRITE_ONCE(x, y) do { *(volatile __typeof__(x)*)&(x) = (y); } while (0)

/* Printf family of functions. */

int vsnprintf(char* buf, size_t buf_size, const char* fmt, va_list ap)
    __attribute__((format(printf, 3, 0)));
int snprintf(char* buf, size_t buf_size, const char* fmt, ...)
    __attribute__((format(printf, 3, 4)));

/* Used by _FORTIFY_SOURCE */
int __vsnprintf_chk(char* buf, size_t buf_size, int flag, size_t real_size, const char* fmt,
                    va_list ap)
    __attribute__((format(printf, 5, 0)));
int __snprintf_chk(char* buf, size_t buf_size, int flag, size_t real_size, const char* fmt, ...)
    __attribute__((format(printf, 5, 6)));

/*
 * Buffered printing. The print_buf structure holds PRINT_BUF_SIZE characters, and outputs them
 * (using `buf_write_all` callback) when `buf_flush()` is called, or when the buffer overflows.
 *
 *     static int buf_write_all(const char* str, size_t size, void* arg) { ... }
 *
 *     struct print_buf buf = INIT_PRINT_BUF(buf_write_all);
 *     buf_puts(&buf, str);
 *     buf_printf(&buf, fmt, ...);
 *     buf_flush(&buf);
 *
 * The `buf_*` functions always return 0, or a negative error code (if one was returned from the
 * `write_all` callback).
 */

#define PRINT_BUF_SIZE 256

struct print_buf {
    char data[PRINT_BUF_SIZE];
    size_t pos;
    void* arg;
    int (*buf_write_all)(const char* str, size_t size, void* arg);
};

#define INIT_PRINT_BUF_ARG(_buf_write_all, _arg) \
    { .pos = 0, .arg = (_arg), .buf_write_all = (_buf_write_all) }
#define INIT_PRINT_BUF(_buf_write_all) \
    { .pos = 0, .arg = NULL, .buf_write_all = (_buf_write_all) }

int buf_vprintf(struct print_buf* buf, const char* fmt, va_list ap)
    __attribute__((format(printf, 2, 0)));
int buf_printf(struct print_buf* buf, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

int buf_puts(struct print_buf* buf, const char* str);
int buf_putc(struct print_buf* buf, char c);
int buf_flush(struct print_buf* buf);

/* Miscelleneous */

int inet_pton4(const char* src, size_t len, void* dst);
int inet_pton6(const char* src, size_t len, void* dst);

uint32_t __htonl(uint32_t x);
uint32_t __ntohl(uint32_t x);
uint16_t __htons(uint16_t x);
uint16_t __ntohs(uint16_t x);

extern const char* const* sys_errlist_internal;

/* Gramine functions */

int get_norm_path(const char* path, char* buf, size_t* inout_size);
int get_base_name(const char* path, char* buf, size_t* inout_size);

/*!
 * \brief Parse a size (number with optional "G"/"M"/"K" suffix) into an uint64_t.
 *
 * \param      str      A string containing a non-negative, decimal number. The string may end with
 *                      "G"/"g" suffix denoting value in GBs, "M"/"m" for MBs, or "K"/"k" for KBs.
 * \param[out] out_val  Parsed size (in bytes).
 *
 * \returns 0 on success, negative if string cannot be parsed into a size (e.g., suffix is wrong).
 */
int parse_size_str(const char* str, uint64_t* out_val);

#define URI_PREFIX_SEPARATOR ":"

#define URI_TYPE_DIR      "dir"
#define URI_TYPE_TCP      "tcp"
#define URI_TYPE_TCP_SRV  "tcp.srv"
#define URI_TYPE_UDP      "udp"
#define URI_TYPE_UDP_SRV  "udp.srv"
#define URI_TYPE_PIPE     "pipe"
#define URI_TYPE_PIPE_SRV "pipe.srv"
#define URI_TYPE_DEV      "dev"
#define URI_TYPE_EVENTFD  "eventfd"
#define URI_TYPE_FILE     "file"

#define URI_PREFIX_DIR      URI_TYPE_DIR URI_PREFIX_SEPARATOR
#define URI_PREFIX_TCP      URI_TYPE_TCP URI_PREFIX_SEPARATOR
#define URI_PREFIX_TCP_SRV  URI_TYPE_TCP_SRV URI_PREFIX_SEPARATOR
#define URI_PREFIX_UDP      URI_TYPE_UDP URI_PREFIX_SEPARATOR
#define URI_PREFIX_UDP_SRV  URI_TYPE_UDP_SRV URI_PREFIX_SEPARATOR
#define URI_PREFIX_PIPE     URI_TYPE_PIPE URI_PREFIX_SEPARATOR
#define URI_PREFIX_PIPE_SRV URI_TYPE_PIPE_SRV URI_PREFIX_SEPARATOR
#define URI_PREFIX_DEV      URI_TYPE_DEV URI_PREFIX_SEPARATOR
#define URI_PREFIX_EVENTFD  URI_TYPE_EVENTFD URI_PREFIX_SEPARATOR
#define URI_PREFIX_FILE     URI_TYPE_FILE URI_PREFIX_SEPARATOR

#define URI_PREFIX_FILE_LEN (static_strlen(URI_PREFIX_FILE))

#define TIME_US_IN_S 1000000ul
#define TIME_US_IN_MS 1000ul
#define TIME_NS_IN_US 1000ul
#define TIME_NS_IN_S (TIME_NS_IN_US * TIME_US_IN_S)

/* Scrub sensitive memory bufs (memset can be optimized away and memset_s is not available in PAL).
 * FIXME: This implementation is inefficient (and used in perf-critical functions).
 * TODO:  Is this really needed? Intel SGX SDK uses similar function as "defense in depth". */
static inline void erase_memory(void* buffer, size_t size) {
    volatile unsigned char* p = buffer;
    while (size--)
        *p++ = 0;
}

#ifdef __x86_64__
static inline bool __range_not_ok(uintptr_t addr, size_t size) {
    addr += size;
    if (addr < size) {
        /* pointer arithmetic overflow, this check is x86-64 specific */
        return true;
    }
    if ((addr & ~(PAGE_SIZE - 1)) == ~(PAGE_SIZE - 1)) {
        /* Disallow the very last page of memory. In C it's legal to have a pointer to the byte
         * after an object (end), yet that would wrap the pointer, which would be wrong. Also it
         * could be dangerous to map stuff there. */
        return true;
    }
    return false;
}

/* Check if pointer to memory region is valid. Return true if the memory
 * region may be valid, false if it is definitely invalid. */
static inline bool access_ok(const volatile void* addr, size_t size) {
    return !__range_not_ok((uintptr_t)addr, size);
}

#else
#error "Unsupported architecture"
#endif /* __x86_64__ */

#if !defined(USE_STDLIB) && __USE_FORTIFY_LEVEL > 0
# include "api_fortified.h"
#endif

#endif /* API_H */
