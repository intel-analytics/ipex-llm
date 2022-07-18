# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2021 Invisible Things Lab
#                    Michał Kowalczyk <mkow@invisiblethingslab.com>

# Commands for temporarily changing source language. Used by other Gramine GDB scripts to ensure
# that a specific source language is used for parsing expressions - GDB interprets scripts using
# the language taken from currently executing code, which may change in time, resulting in scripts
# working only part of the time.
#
# Example: When a GDB script does `if *(uint16_t*)$rip == 0x1234` in a signal catchpoint then it
# will fail with a syntax error if the binary debugged is written in Rust, but only if the signal
# arrived while executing Rust code.

import re

import gdb # pylint: disable=import-error

_g_languages = []


class PushLanguage(gdb.Command):
    """Temporarily change source language and save the old one"""

    def __init__(self):
        super().__init__('push-language', gdb.COMMAND_USER)

    def invoke(self, arg, _from_tty):
        self.dont_repeat()

        lang_str = gdb.execute('show language', to_string=True).strip()
        # ';' is for things like: "auto; currently c"
        m = re.match(r'The current source language is "(.*?)[";]', lang_str)
        assert m, 'Unexpected output from \'show language\': ' + lang_str
        _g_languages.append(m.group(1))

        gdb.execute('set language ' + arg)


class PopLanguage(gdb.Command):
    """Recover source language saved by PushLanguage"""

    def __init__(self):
        super().__init__('pop-language', gdb.COMMAND_USER)

    def invoke(self, arg, _from_tty):
        self.dont_repeat()

        assert arg == ''
        lang = _g_languages.pop()
        gdb.execute('set language ' + lang)


def main():
    PushLanguage()
    PopLanguage()


if __name__ == '__main__':
    main()
