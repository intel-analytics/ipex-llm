/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Labs */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    char* secret = getenv("SECRET_PROVISION_SECRET_STRING");
    if (secret)
        printf("--- Received secret = '%s' ---\n", secret);
    else
        printf("--- Did not receive any secret! ---\n");

    return 0;
}
