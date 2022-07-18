/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#include "dump.h"

int main(int argc, char** argv) {
    if (dump_path("/sys") < 0)
        return 1;
    return 0;
}
