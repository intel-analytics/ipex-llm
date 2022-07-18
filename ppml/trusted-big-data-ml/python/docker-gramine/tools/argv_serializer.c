/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 */

/* Helper tool for protected argv ("loader.argv_src_file" manifest option). See Gramine
 * documentation for usage.
 */

#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++)
        if (fwrite(argv[i], strlen(argv[i]) + 1, 1, stdout) != 1)
            return 1;
    return 0;
}
