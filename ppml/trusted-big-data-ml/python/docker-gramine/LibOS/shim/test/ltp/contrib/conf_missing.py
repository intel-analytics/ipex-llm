#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2019  Wojtek Porczyk <woju@invisiblethingslab.com>

import argparse
import configparser

argparser = argparse.ArgumentParser()
argparser.add_argument('--config', '-c', metavar='FILENAME',
    type=argparse.FileType('r'),
    help='location of ltp.cfg file')
argparser.add_argument('file', metavar='FILENAME',
    type=argparse.FileType('r'), nargs='?', default='-',
    help='LTP scenario file')

def main(args=None):
    args = argparser.parse_args(args)
    config = configparser.ConfigParser()
    config.read_file(args.config)

    with args.file:
        for line in args.file:
            line = line.strip()
            if not line or line[0] == '#':
                continue

            tag, cmd = line.split(maxsplit=1)
            if not tag in config and not any(c in cmd for c in '|;&'):
                print(f'[{tag}]\nmust-pass =\n')

if __name__ == '__main__':
    main()
