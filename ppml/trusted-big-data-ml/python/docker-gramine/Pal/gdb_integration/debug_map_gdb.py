# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2020 Intel Corporation
#                    Paweł Marczewski <pawel@invisiblethingslab.com>

# Debug map handling, so that GDB sees all ELF binaries loaded by Gramine. Connects with
# debug_map.c (in PAL) using a breakpoint on debug_map_update_debugger() function.

import os
import shlex

import gdb  # pylint: disable=import-error

try:
    from elftools.elf.elffile import ELFFile
except ImportError:
    print('Python elftools module not found, please install (e.g. apt install python3-pyelftools)')
    raise

# TODO (GDB 8.2): We need to load the ELF file and determine the section addresses manually, because
# GDB before version 8.2 needs us to provide text address, and all the other section addresses, when
# loading a file:
#
#     add-symbol-file <file_name> <text_addr> -s <section> <section_addr>...
#
# GDB 8.2 makes the text_addr parameter optional, and adds an '-o <load_addr>' parameter, which is
# enough for GDB to load all the sections. When we can depend on it, we will be able to stop parsing
# the ELF file here.

def load_elf_sections(file_name, load_addr):
    '''
    Open an ELF file and determine a list of sections along with addresses.

    Returns a list of (name, addr) elements.
    '''

    if not os.path.exists(file_name):
        print('file not found: {}'.format(file_name))
        return {}

    sections = []
    with open(file_name, 'rb') as f:
        elf = ELFFile(f)

        for section in elf.iter_sections():
            if section.name and section.header['sh_addr']:
                # Workaround for old version of pyelftools (Ubuntu 16)
                # that stores section.name as bytes.
                name = section.name
                if isinstance(name, bytes):
                    name = name.decode('ascii')

                addr = load_addr + section.header['sh_addr']
                sections.append((name, addr))

    return sections


def retrieve_debug_maps():
    '''
    Retrieve the debug_map structure from the inferior process. The result is a dict with the
    following structure:

    {load_addr: (file_name, text_addr, [(name, addr)])}
    '''

    debug_maps = {}
    val_map = gdb.parse_and_eval('g_debug_map')
    while int(val_map) != 0:
        file_name = val_map['name'].string()
        load_addr = int(val_map['addr'])

        if file_name.startswith('['):
            # This is vDSO, not a real file.
            debug_maps[load_addr] = (file_name, load_addr, [])
        else:
            file_name = os.path.abspath(file_name)
            sections = load_elf_sections(file_name, load_addr)
            text_addr = None
            for name, addr in sections:
                if name == '.text':
                    text_addr = addr
                    break
            # We need the text_addr to use add-symbol-file (at least until GDB 8.2).
            if text_addr is not None:
                debug_maps[load_addr] = (file_name, text_addr, sections)

        val_map = val_map['next']

    return debug_maps


class UpdateDebugMaps(gdb.Command):
    """Update debug maps for the inferior process."""

    def __init__(self):
        super().__init__('update-debug-maps', gdb.COMMAND_USER)

    def invoke(self, arg, _from_tty):
        self.dont_repeat()
        assert arg == ''

        # Store the currently loaded maps inside the Progspace object, so that we can compare
        # old and new states. See:
        # https://sourceware.org/gdb/current/onlinedocs/gdb/Progspaces-In-Python.html
        progspace = gdb.current_progspace()
        if not hasattr(progspace, 'debug_maps'):
            progspace.debug_maps = {}

        old = progspace.debug_maps
        new = retrieve_debug_maps()
        for load_addr in set(old) | set(new):
            # Skip unload/reload if the map is unchanged
            if old.get(load_addr) == new.get(load_addr):
                continue

            if load_addr in old:
                # Log the removing, because remove-symbol-file itself doesn't produce helpful output
                # on errors.
                file_name, text_addr, sections = old[load_addr]
                print("Removing symbol file (was {}) from addr: 0x{:x}".format(
                    file_name, load_addr))
                try:
                    gdb.execute('remove-symbol-file -a 0x{:x}'.format(text_addr))
                except gdb.error as e:
                    print('warning: failed to remove symbol file: {}'.format(e))

            # Note that we escape text arguments to 'add-symbol-file' (file name and section names)
            # using shlex.quote(), because GDB commands use a shell-like argument syntax.

            if load_addr in new:
                file_name, text_addr, sections = new[load_addr]

                if file_name.startswith('['):
                    # This is vDSO, not a real file.
                    cmd = 'add-symbol-file-from-memory 0x{:x}'.format(load_addr)
                else:
                    cmd = 'add-symbol-file {} 0x{:x} '.format(
                        shlex.quote(file_name), text_addr)
                    cmd += ' '.join('-s {} 0x{:x}'.format(shlex.quote(name), addr)
                                    for name, addr in sections
                                    if name != '.text')
                try:
                    # Temporarily disable pagination, because 'add-symbol-file` produces a lot of
                    # noise.
                    gdb.execute('push-pagination off')

                    gdb.execute(cmd)
                finally:
                    gdb.execute('pop-pagination')

        progspace.debug_maps = new


class DebugMapBreakpoint(gdb.Breakpoint):
    def __init__(self):
        gdb.Breakpoint.__init__(self, spec="debug_map_update_debugger", internal=True)

    def stop(self):
        gdb.execute('update-debug-maps')
        # return False to continue automatically after the breakpoint
        return False


def debug_map_stop_handler(event):
    # Make sure we handle connecting to a new process correctly:
    # update the debug maps if we never did it before.
    if not isinstance(event, gdb.BreakpointEvent):
        progspace = gdb.current_progspace()
        if not hasattr(progspace, 'debug_maps'):
            gdb.execute('update-debug-maps')


def debug_map_clear_objfiles_handler(event):
    # Record that symbol files have been cleared on GDB's side (e.g. on program exit), so that we do
    # not try to remove them again.
    if hasattr(event.progspace, 'debug_maps'):
        delattr(event.progspace, 'debug_maps')


def main():
    UpdateDebugMaps()
    DebugMapBreakpoint()

    gdb.events.stop.connect(debug_map_stop_handler)
    gdb.events.clear_objfiles.connect(debug_map_clear_objfiles_handler)


if __name__ == '__main__':
    main()
