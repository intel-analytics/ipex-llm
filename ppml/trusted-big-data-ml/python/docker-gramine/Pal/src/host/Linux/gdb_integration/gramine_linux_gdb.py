# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2020 Intel Corporation
#                    Michał Kowalczyk <mkow@invisiblethingslab.com>

import os
import gdb  # pylint: disable=import-error

def main():
    common_path = '../../gdb_integration/'
    for filename in [
            common_path + 'language_gdb.py',
            common_path + 'pagination_gdb.py',
            common_path + 'debug_map_gdb.py',
            common_path + 'gramine.gdb',
    ]:
        print("[%s] Loading %s..." % (os.path.basename(__file__), filename))
        path = os.path.join(os.path.dirname(__file__), filename)
        gdb.execute("source " + path)

if __name__ == '__main__':
    main()
