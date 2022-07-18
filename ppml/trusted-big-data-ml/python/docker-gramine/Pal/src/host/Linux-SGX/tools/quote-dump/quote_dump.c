/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Rafal Wojdyla <omeg@invisiblethingslab.com>
 */

#define _GNU_SOURCE

#include <getopt.h>
#include <stdio.h>

#include "attestation.h"
#include "util.h"

struct option g_options[] = {
    { "help", no_argument, 0, 'h' },
    { "msb", no_argument, 0, 'm' },
    { 0, 0, 0, 0 }
};

static void usage(const char* exec) {
    INFO("Usage: %s [options] <quote path>\n", exec);
    INFO("Available options:\n");
    INFO("  --help, -h  Display this help\n");
    INFO("  --msb, -m   Display hex strings in big-endian order\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return -1;
    }

    endianness_t endian = ENDIAN_LSB;

    int option = 0;
    // parse command line
    while (true) {
        option = getopt_long(argc, argv, "hm", g_options, NULL);
        if (option == -1)
            break;

        switch (option) {
            case 'h':
                usage(argv[0]);
                return 0;
            case 'm':
                endian = ENDIAN_MSB;
                break;
            default:
                usage(argv[0]);
                return -1;
        }
    }

    if (optind >= argc) {
        ERROR("Quote path not specified\n");
        usage(argv[0]);
        return -1;
    }

    const char* path = argv[optind++];

    size_t quote_size = 0;
    void* quote = read_file(path, &quote_size, /*buffer=*/NULL);
    if (!quote)
        return -1;

    set_endianness(endian);
    display_quote(quote, quote_size);
    return 0;
}
