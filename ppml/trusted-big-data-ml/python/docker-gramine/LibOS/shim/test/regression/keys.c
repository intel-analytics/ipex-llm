/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/* Test for setting and reading encrypted files keys (/dev/attestation/keys). */

#include <assert.h>
#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "rw_file.h"

#define KEY_SIZE 16
#define KEY_STR_SIZE (2 * KEY_SIZE + 1)

typedef uint8_t pf_key_t[KEY_SIZE];

/* Keys and values specified in `manifest.template`. */
#define DEFAULT_KEY_PATH "/dev/attestation/keys/default"
#define CUSTOM_KEY_PATH "/dev/attestation/keys/my_custom_key"

static const pf_key_t default_key = {
    0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00};

static const pf_key_t custom_key = {
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};

/* New value that we'll be setting. */
static const pf_key_t new_custom_key = {
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};

static void format_key(char buf[static KEY_STR_SIZE], const pf_key_t* key) {
    const char* hex = "0123456789abcdef";

    for (size_t i = 0; i < KEY_SIZE; i++) {
        uint8_t byte = (*key)[i];
        buf[i * 2]     = hex[byte / 16];
        buf[i * 2 + 1] = hex[byte % 16];
    }
    buf[KEY_SIZE * 2] = '\0';
}

static void expect_key(const char* desc, const char* path, const pf_key_t* expected_key) {
    pf_key_t key;

    ssize_t n = posix_file_read(path, (char*)&key, sizeof(key));
    if (n < 0)
        err(1, "%s: error reading %s", desc, path);
    if ((size_t)n < sizeof(key))
        errx(1, "%s: file %s is too short: %zd", desc, path, n);

    if (memcmp(&key, expected_key, sizeof(*expected_key))) {
        char expected_key_str[KEY_STR_SIZE];
        format_key(expected_key_str, expected_key);

        char key_str[KEY_STR_SIZE];
        format_key(key_str, &key);

        errx(1, "%s: wrong key: expected %s, got %s", desc, expected_key_str, key_str);
    }
}

static void write_key(const char* desc, const char* path, const pf_key_t* key) {
    ssize_t n = posix_file_write(path, (char*)key, sizeof(*key));
    if (n < 0)
        err(1, "%s: error writing %s", desc, path);
    if ((size_t)n < sizeof(*key))
        errx(1, "%s: not enough bytes written to %s: %zd", desc, path, n);
}

int main(void) {
    expect_key("before writing key", DEFAULT_KEY_PATH, &default_key);
    expect_key("before writing key", CUSTOM_KEY_PATH, &custom_key);

    /* Perform invalid write (size too small), Gramine's `write` syscall should fail */
    ssize_t n = posix_file_write(CUSTOM_KEY_PATH, (char*)&new_custom_key,
                                 sizeof(new_custom_key) - 1);
    if (n >= 0 || (n < 0 && errno != EACCES))
        err(1, "writing invalid key: expected EACCES");

    expect_key("after writing invalid key", CUSTOM_KEY_PATH, &custom_key);

    write_key("writing key", CUSTOM_KEY_PATH, &new_custom_key);
    expect_key("after writing key", CUSTOM_KEY_PATH, &new_custom_key);

    /* Check if the child process will see the updated key. */
    pid_t pid = fork();
    if (pid < 0)
        err(1, "fork");
    if (pid == 0) {
        expect_key("in child process", DEFAULT_KEY_PATH, &default_key);
        expect_key("in child process", CUSTOM_KEY_PATH, &new_custom_key);
    } else {
        int status;
        if (waitpid(pid, &status, 0) == -1)
            err(1, "waitpid");
        if (!WIFEXITED(status))
            errx(1, "child not exited");
        if (WEXITSTATUS(status) != 0)
            errx(1, "unexpected exit status: %d", WEXITSTATUS(status));
        printf("TEST OK\n");
    }

    return 0;
}
