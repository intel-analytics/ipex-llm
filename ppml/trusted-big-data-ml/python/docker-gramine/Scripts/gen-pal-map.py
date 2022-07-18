#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2021 Intel Corporation
#                    Wojtek Porczyk <woju@invisiblethingslab.com>
#

# TODO: after upgrading to meson 0.57, replace this with module('fs').read() + configure_file()

import argparse
import string

class MesonTemplate(string.Template):
    pattern = '''
        @(?:
            (?P<escaped>@) |
            (?P<named>[A-Za-z0-9_]+)@ |
            (?P<braced>[A-Za-z0-9_]+)@ |
            (?P<invalid>)
        )
    '''

argparser = argparse.ArgumentParser()
argparser.add_argument('pal_symbols', type=argparse.FileType('r'))
argparser.add_argument('infile', type=argparse.FileType('r'))
argparser.add_argument('outfile', type=argparse.FileType('w'))

def main(args=None):
    args = argparser.parse_args(args)
    pal_symbols = ' '.join(f'{sym};' for sym in args.pal_symbols.read().strip().split())
    template = MesonTemplate(args.infile.read())
    args.outfile.write(template.substitute(PAL_SYMBOLS=pal_symbols))

if __name__ == '__main__':
    main()
