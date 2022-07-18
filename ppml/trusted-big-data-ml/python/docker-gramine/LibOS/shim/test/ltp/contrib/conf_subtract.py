#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2020  Intel Corporation
#                     Paweł Marczewski <pawel@invisiblethingslab.com>

import argparse

DEFAULT = 'DEFAULT'

# Compute a difference between two files.

# textual config representation to preserve comments

argparser = argparse.ArgumentParser()
argparser.add_argument('file_a', metavar='FILE_A',
    type=argparse.FileType('r'),
    help='first file')

argparser.add_argument('file_b', metavar='FILE_B',
    type=argparse.FileType('r'),
    help='second file')

def read_file(file):
    sections = {}

    section = ''
    section_name = None
    for line in file:
        if line.startswith('['):
            if section_name is None:
                # We're in a new section already, this is its name.
                section_name = line.strip(' \n[]')
                section += line
            else:
                # We start a new section.
                sections[section_name] = section
                section_name = line.strip(' \n[]')
                section = line
        elif line.startswith('#') and section_name is not None:
            # Treat comments as a start of a new section.
            sections[section_name] = section

            section = line
            section_name = None
        else:
            section += line

    assert section_name is not None
    sections[section_name] = section
    return sections


def main(args=None):
    args = argparser.parse_args(args)

    sections_a = read_file(args.file_a)
    sections_b = read_file(args.file_b)

    for name, section in sections_b.items():
        if name not in sections_a or sections_a[name] != sections_b[name]:
            print(section, end='')

if __name__ == '__main__':
    main()
