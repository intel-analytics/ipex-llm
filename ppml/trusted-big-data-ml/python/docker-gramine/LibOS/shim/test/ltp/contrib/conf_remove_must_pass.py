#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2019  Wojtek Porczyk <woju@invisiblethingslab.com>

import argparse
import enum
import sys

# textual config representation to preserve comments

argparser = argparse.ArgumentParser()
argparser.add_argument('--config', '-c', metavar='FILENAME',
    type=argparse.FileType('r'),
    default='-',
    help='location of ltp.cfg file')
argparser.add_argument('sections', metavar='SECTION',
    nargs='+',
    help='sections to be removed')

class State(enum.Enum):
    IDLE, ACCUMULATING, DROPPING = range(3)

def flush(accumulator):
    if accumulator and any(i.strip() for i in accumulator[1:]):
        for i in accumulator:
            sys.stdout.write(i)
    accumulator.clear()

def main(args=None):
    args = argparser.parse_args(args)

    with args.config as file:
        state = State.IDLE
        accumulator = []

        for line in file:
            if line.lstrip().startswith('['):
                section = line.strip(' \n[]')

                flush(accumulator)

                if section in args.sections:
                    state = State.ACCUMULATING
                else:
                    state = State.IDLE

            elif line.startswith('must-pass') and state is State.ACCUMULATING:
                state = State.DROPPING

            elif line[0] in ' \t':
                pass

            elif state is State.DROPPING:
                state = State.ACCUMULATING


            if state is State.IDLE:
                sys.stdout.write(line)
            elif state is State.ACCUMULATING:
                accumulator.append(line)

        flush(accumulator)


if __name__ == '__main__':
    main()
