# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2020 Intel Corporation
#                    Michał Kowalczyk <mkow@invisiblethingslab.com>
#                    Paweł Marczewski <pawel@invisiblethingslab.com>

# Commands for temporarily changing pagination state. Used by other Gramine GDB scripts in
# situation where we produce a lot of noise and we don't want to repeatedly prompt user for
# continuation.

import gdb # pylint: disable=import-error

_g_paginations = []


class PushPagination(gdb.Command):
    """Temporarily change pagination and save the old state"""

    def __init__(self):
        super().__init__('push-pagination', gdb.COMMAND_USER)

    def invoke(self, arg, _from_tty):
        self.dont_repeat()

        pagination_str = gdb.execute('show pagination', to_string=True).strip()
        assert pagination_str in ('State of pagination is on.', 'State of pagination is off.')
        pagination = pagination_str.endswith('on.')
        _g_paginations.append(pagination)

        assert arg in ('on', 'off')
        gdb.execute('set pagination ' + arg)


class PopPagination(gdb.Command):
    """Recover pagination state saved by PushPagination"""

    def __init__(self):
        super().__init__('pop-pagination', gdb.COMMAND_USER)

    def invoke(self, arg, _from_tty):
        self.dont_repeat()

        assert arg == ''
        pagination = _g_paginations.pop()
        gdb.execute('set pagination ' + ('on' if pagination else 'off'))


def main():
    PushPagination()
    PopPagination()


if __name__ == '__main__':
    main()
