#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2019  Wojtek Porczyk <woju@invisiblethingslab.com>

'''
Check ltp.cfg for tag sorting
'''

import argparse
import sys
import fnmatch
import shlex

argparser = argparse.ArgumentParser()
argparser.add_argument('config', metavar='FILENAME',
    type=argparse.FileType('r'), nargs='+', default='-',
    help='ltp.cfg file (default: stdin)')

argparser.add_argument('--scenario', metavar='FILENAME',
    type=argparse.FileType('r'),
    help='list of tests')


def read_sections(file):
    with file:
        for i, line in enumerate(file, 1):
            line = line.strip()
            if not line.startswith('['):
                continue
            yield i, line.strip(' []')


def read_tags(file):
    with file:
        for line in file:
            if line[0] in '\n#':
                continue
            yield shlex.split(line)[0]


def validate_section_order(name, sections):
    mistakes = 0
    prev = ''
    for lineno, section in sections:
        if section == prev:
            print('{name}:{lineno}: duplicate section: [{section}]'.format(
                name=name, lineno=lineno, section=section))
            mistakes += 1
        if section < prev:
            print('{name}:{lineno}: bad order: [{section}] (after [{prev}])'.format(
                name=name, lineno=lineno, section=section, prev=prev))
            mistakes += 1
        prev = section
    return mistakes


def validate_section_names(name, sections, tags):
    mistakes = 0

    for lineno, section in sections:
        if set(section) & set('*?[]!'):  # fnmatch pattern
            if not any(tag for tag in tags if fnmatch.fnmatch(tag, section)):
                print("{name}:{lineno}: pattern doesn't match any test: [{section}]".format(
                    name=name, lineno=lineno, section=section))
                mistakes += 1
        elif section != 'DEFAULT':
            if section not in tags:
                print("{name}:{lineno}: test doesn't exist: [{section}]".format(
                    name=name, lineno=lineno, section=section))
                mistakes += 1
    return mistakes


def main(args=None):
    args = argparser.parse_args(args)
    tags = set(read_tags(args.scenario)) if args.scenario else None
    for config in args.config:
        sections = list(read_sections(config))

        mistakes = 0
        mistakes += validate_section_order(config.name, sections)
        if args.scenario:
            mistakes += validate_section_names(config.name, sections, tags)

    return min(mistakes, 255)

if __name__ == '__main__':
    sys.exit(main())
