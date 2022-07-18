#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2019  Wojtek Porczyk <woju@invisiblethingslab.com>

import argparse
import collections

DEFAULT = 'DEFAULT'

# textual config representation to preserve comments

argparser = argparse.ArgumentParser()
argparser.add_argument('files', metavar='FILENAME',
    type=argparse.FileType('r'),
    nargs='+',
    help='.cfg files to be merged')

def print_section(section, lines):
    if section is not None:
        print(f'[{section}]')
    for line in lines:
        print(line, end='')

def main(args=None):
    args = argparser.parse_args(args)
    sections = collections.defaultdict(list)

    for file in args.files:
        section = None

        with file:
            for line in file:
                if line.lstrip().startswith('['):
                    section = line.strip(' \n[]')
                else:
                    sections[section].append(line)

    for section in (None, DEFAULT):
        if section not in sections:
            continue
        print_section(section, sections.pop(section))

    for section in sorted(sections):
        print_section(section, sections[section])

if __name__ == '__main__':
    main()
