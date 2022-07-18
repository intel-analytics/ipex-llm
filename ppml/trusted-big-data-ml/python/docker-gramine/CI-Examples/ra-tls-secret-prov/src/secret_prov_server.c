/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Labs */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "secret_prov.h"

#define EXPECTED_STRING "MORE"
#define SECRET_STRING "42" /* answer to ultimate question of life, universe, and everything */

#define WRAP_KEY_FILENAME "files/wrap-key"
#define WRAP_KEY_SIZE     16

#define SRV_CRT_PATH "ssl/server.crt"
#define SRV_KEY_PATH "ssl/server.key"

static pthread_mutex_t g_print_lock;
char g_secret_pf_key_hex[WRAP_KEY_SIZE * 2 + 1];

static void hexdump_mem(const void* data, size_t size) {
    uint8_t* ptr = (uint8_t*)data;
    for (size_t i = 0; i < size; i++)
        printf("%02x", ptr[i]);
    printf("\n");
}

/* our own callback to verify SGX measurements during TLS handshake */
static int verify_measurements_callback(const char* mrenclave, const char* mrsigner,
                                        const char* isv_prod_id, const char* isv_svn) {
    assert(mrenclave && mrsigner && isv_prod_id && isv_svn);

    pthread_mutex_lock(&g_print_lock);
    puts("Received the following measurements from the client:");
    printf("  - MRENCLAVE:   "); hexdump_mem(mrenclave, 32);
    printf("  - MRSIGNER:    "); hexdump_mem(mrsigner, 32);
    printf("  - ISV_PROD_ID: %hu\n", *((uint16_t*)isv_prod_id));
    printf("  - ISV_SVN:     %hu\n", *((uint16_t*)isv_svn));
    puts("[ WARNING: In reality, you would want to compare against expected values! ]");
    pthread_mutex_unlock(&g_print_lock);

    return 0;
}

/* this callback is called in a new thread associated with a client; be careful to make this code
 * thread-local and/or thread-safe */
static int communicate_with_client_callback(struct ra_tls_ctx* ctx) {
    int ret;

    /* if we reached this callback, the first secret was sent successfully */
    printf("--- Sent secret1 = '%s' ---\n", g_secret_pf_key_hex);

    /* let's send another secret (just to show communication with secret-awaiting client) */
    int bytes;
    uint8_t buf[128] = {0};

    bytes = secret_provision_read(ctx, buf, sizeof(EXPECTED_STRING));
    if (bytes < 0) {
        if (bytes == -ECONNRESET) {
            /* client doesn't want another secret, shutdown communication gracefully */
            ret = 0;
            goto out;
        }

        fprintf(stderr, "[error] secret_provision_read() returned %d\n", bytes);
        ret = -EINVAL;
        goto out;
    }

    assert(bytes == sizeof(EXPECTED_STRING));
    if (memcmp(buf, EXPECTED_STRING, bytes)) {
        fprintf(stderr, "[error] client sent '%s' but expected '%s'\n", buf, EXPECTED_STRING);
        ret = -EINVAL;
        goto out;
    }

    bytes = secret_provision_write(ctx, (uint8_t*)SECRET_STRING, sizeof(SECRET_STRING));
    if (bytes < 0) {
        fprintf(stderr, "[error] secret_provision_write() returned %d\n", bytes);
        ret = -EINVAL;
        goto out;
    }

    printf("--- Sent secret2 = '%s' ---\n", SECRET_STRING);
    ret = 0;
out:
    secret_provision_close(ctx);
    return ret;
}

int main(int argc, char** argv) {
    int ret;

    ret = pthread_mutex_init(&g_print_lock, NULL);
    if (ret < 0)
        return ret;

    puts("--- Reading the master key for protected files from '" WRAP_KEY_FILENAME "' ---");
    int fd = open(WRAP_KEY_FILENAME, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[error] cannot open '" WRAP_KEY_FILENAME "'\n");
        return 1;
    }

    char buf[WRAP_KEY_SIZE + 1] = {0}; /* +1 is to detect if file is not bigger than expected */
    ssize_t bytes_read = 0;
    while (1) {
        ssize_t ret = read(fd, buf + bytes_read, sizeof(buf) - bytes_read);
        if (ret > 0) {
            bytes_read += ret;
        } else if (ret == 0) {
            /* end of file */
            break;
        } else if (errno == EAGAIN || errno == EINTR) {
            continue;
        } else {
            fprintf(stderr, "[error] cannot read '" WRAP_KEY_FILENAME "'\n");
            close(fd);
            return 1;
        }
    }

    ret = close(fd);
    if (ret < 0) {
        fprintf(stderr, "[error] cannot close '" WRAP_KEY_FILENAME "'\n");
        return 1;
    }

    if (bytes_read != WRAP_KEY_SIZE) {
        fprintf(stderr, "[error] encryption key from '" WRAP_KEY_FILENAME "' is not 16B in size\n");
        return 1;
    }

    uint8_t* ptr = (uint8_t*)buf;
    for (size_t i = 0; i < bytes_read; i++)
        sprintf(&g_secret_pf_key_hex[i * 2], "%02x", ptr[i]);

    puts("--- Starting the Secret Provisioning server on port 4433 ---");
    ret = secret_provision_start_server((uint8_t*)g_secret_pf_key_hex, sizeof(g_secret_pf_key_hex),
                                        "4433", SRV_CRT_PATH, SRV_KEY_PATH,
                                        verify_measurements_callback,
                                        communicate_with_client_callback);
    if (ret < 0) {
        fprintf(stderr, "[error] secret_provision_start_server() returned %d\n", ret);
        return 1;
    }

    pthread_mutex_destroy(&g_print_lock);
    return 0;
}
