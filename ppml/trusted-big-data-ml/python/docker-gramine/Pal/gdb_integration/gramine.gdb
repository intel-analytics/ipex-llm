# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2021 Intel Corporation
#                    Michał Kowalczyk <mkow@invisiblethingslab.com>
#                    Paweł Marczewski <pawel@invisiblethingslab.com>

# Gramine GDB configuration (common for all hosts).

# Used internally by Gramine, generates a lot of noise if we don't silence it.
handle SIGCONT pass noprint nostop

# Disable loading inferior-specific libthread_db library. This does not work with our patched
# libpthread, and prevents newer GDB versions (9.2+) from working when a program uses libpthread.
set auto-load libthread-db off
set libthread-db-search-path ""

# Reenable address space layout randomization (ASLR). Gramine's features often take memory layout
# into account, so running with ASLR enabled is more realistic and allows us to catch issues sooner.
#
# Note that Linux PAL disables ASLR on startup anyway, so this setting is irrelevant for Linux PAL.
set disable-randomization off

# Make GDB follow both sides of the fork - GDB (at least version 8.1) crashes on Gramine running
# some applications otherwise (e.g. exit_group regression test).
set detach-on-fork off

# Resume all processes by default. This is to negate consequences of 'set detach-on-fork off': by
# default, it keeps running only one side of the fork. This is usually not what we want, and it's
# particularly annoying in Gramine, because checkpoint data is sent between processes just after
# forking.
set schedule-multiple on
