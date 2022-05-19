#!/usr/bin/env python
# pep8.py - Check Python source code formatting, according to PEP 8
# Copyright (C) 2006-2009 Johann C. Rocholl <johann@rocholl.net>
# Copyright (C) 2009-2014 Florent Xicluna <florent.xicluna@gmail.com>
# Copyright (C) 2014-2016 Ian Lee <ianlee1521@gmail.com>
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

r"""
Check Python source code formatting, according to PEP 8.

For usage and a list of options, try this:
$ python pep8.py -h

This program and its regression test suite live here:
https://github.com/pycqa/pep8

Groups of errors and warnings:
E errors
W warnings
100 indentation
200 whitespace
300 blank lines
400 imports
500 line length
600 deprecation
700 statements
900 syntax error
"""
from __future__ import with_statement

import os
import sys
import re
import time
import inspect
import keyword
import tokenize
from optparse import OptionParser
from fnmatch import fnmatch
try:
    from configparser import RawConfigParser
    from io import TextIOWrapper
except ImportError:
    from ConfigParser import RawConfigParser

__version__ = '1.7.0'

DEFAULT_EXCLUDE = '.svn,CVS,.bzr,.hg,.git,__pycache__,.tox'
DEFAULT_IGNORE = 'E121,E123,E126,E226,E24,E704'
try:
    if sys.platform == 'win32':
        USER_CONFIG = os.path.expanduser(r'~\.pep8')
    else:
        USER_CONFIG = os.path.join(
            os.getenv('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'),
            'pep8'
        )
except ImportError:
    USER_CONFIG = None

PROJECT_CONFIG = ('setup.cfg', 'tox.ini', '.pep8')
TESTSUITE_PATH = os.path.join(os.path.dirname(__file__), 'testsuite')
MAX_LINE_LENGTH = 100
REPORT_FORMAT = {
    'default': '%(path)s:%(row)d:%(col)d: %(code)s %(text)s',
    'pylint': '%(path)s:%(row)d: [%(code)s] %(text)s',
}

PyCF_ONLY_AST = 1024
SINGLETONS = frozenset(['False', 'None', 'True'])
KEYWORDS = frozenset(keyword.kwlist + ['print']) - SINGLETONS
UNARY_OPERATORS = frozenset(['>>', '**', '*', '+', '-'])
ARITHMETIC_OP = frozenset(['**', '*', '/', '//', '+', '-'])
WS_OPTIONAL_OPERATORS = ARITHMETIC_OP.union(['^', '&', '|', '<<', '>>', '%'])
WS_NEEDED_OPERATORS = frozenset([
    '**=', '*=', '/=', '//=', '+=', '-=', '!=', '<>', '<', '>',
    '%=', '^=', '&=', '|=', '==', '<=', '>=', '<<=', '>>=', '='])
WHITESPACE = frozenset(' \t')
NEWLINE = frozenset([tokenize.NL, tokenize.NEWLINE])
SKIP_TOKENS = NEWLINE.union([tokenize.INDENT, tokenize.DEDENT])
# ERRORTOKEN is triggered by backticks in Python 3
SKIP_COMMENTS = SKIP_TOKENS.union([tokenize.COMMENT, tokenize.ERRORTOKEN])
BENCHMARK_KEYS = ['directories', 'files', 'logical lines', 'physical lines']

INDENT_REGEX = re.compile(r'([ \t]*)')
RAISE_COMMA_REGEX = re.compile(r'raise\s+\w+\s*,')
RERAISE_COMMA_REGEX = re.compile(r'raise\s+\w+\s*,.*,\s*\w+\s*$')
ERRORCODE_REGEX = re.compile(r'\b[A-Z]\d{3}\b')
DOCSTRING_REGEX = re.compile(r'u?r?["\']')
EXTRANEOUS_WHITESPACE_REGEX = re.compile(r'[\[({] | [\]}),;:]')
WHITESPACE_AFTER_COMMA_REGEX = re.compile(r'[,;:]\s*(?:  |\t)')
COMPARE_SINGLETON_REGEX = re.compile(r'(\bNone|\bFalse|\bTrue)?\s*([=!]=)'
                                     r'\s*(?(1)|(None|False|True))\b')
COMPARE_NEGATIVE_REGEX = re.compile(r'\b(not)\s+[^][)(}{ ]+\s+(in|is)\s')
COMPARE_TYPE_REGEX = re.compile(r'(?:[=!]=|is(?:\s+not)?)\s*type(?:s.\w+Type'
                                r'|\s*\(\s*([^)]*[^ )])\s*\))')
KEYWORD_REGEX = re.compile(r'(\s*)\b(?:%s)\b(\s*)' % r'|'.join(KEYWORDS))
OPERATOR_REGEX = re.compile(r'(?:[^,\s])(\s*)(?:[-+*/|!<=>%&^]+)(\s*)')
LAMBDA_REGEX = re.compile(r'\blambda\b')
HUNK_REGEX = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@.*$')

# Work around Python < 2.6 behaviour, which does not generate NL after
# a comment which is on a line by itself.
COMMENT_WITH_NL = tokenize.generate_tokens(['#\n'].pop).send(None)[1] == '#\n'


##############################################################################
# Plugins (check functions) for physical lines
##############################################################################

def trailing_blacklist_words(physical_line):
    if physical_line.find("assert ") != -1:
        return 0, "Please don't use assert, use log4Error.invalidInputError instead"
    if physical_line.find("raise ") != -1:
        return 0, "Please don't use raise, use log4Error.invalidInputError instead"



##############################################################################
# Helper functions
##############################################################################


if sys.version_info < (3,):
    # Python 2: implicit encoding.
    def readlines(filename):
        """Read the source code."""
        with open(filename, 'rU') as f:
            return f.readlines()
    isidentifier = re.compile(r'[a-zA-Z_]\w*$').match
    stdin_get_value = sys.stdin.read
else:
    # Python 3
    def readlines(filename):
        """Read the source code."""
        try:
            with open(filename, 'rb') as f:
                (coding, lines) = tokenize.detect_encoding(f.readline)
                f = TextIOWrapper(f, coding, line_buffering=True)
                return [l.decode(coding) for l in lines] + f.readlines()
        except (LookupError, SyntaxError, UnicodeError):
            # Fall back if file encoding is improperly declared
            with open(filename, encoding='latin-1') as f:
                return f.readlines()
    isidentifier = str.isidentifier

    def stdin_get_value():
        return TextIOWrapper(sys.stdin.buffer, errors='ignore').read()
noqa = re.compile(r'# no(?:qa|pep8)\b', re.I).search


def expand_indent(line):
    r"""Return the amount of indentation.

    Tabs are expanded to the next multiple of 8.

    >>> expand_indent('    ')
    4
    >>> expand_indent('\t')
    8
    >>> expand_indent('       \t')
    8
    >>> expand_indent('        \t')
    16
    """
    if '\t' not in line:
        return len(line) - len(line.lstrip())
    result = 0
    for char in line:
        if char == '\t':
            result = result // 8 * 8 + 8
        elif char == ' ':
            result += 1
        else:
            break
    return result


def mute_string(text):
    """Replace contents with 'xxx' to prevent syntax matching.

    >>> mute_string('"abc"')
    '"xxx"'
    >>> mute_string("'''abc'''")
    "'''xxx'''"
    >>> mute_string("r'abc'")
    "r'xxx'"
    """
    # String modifiers (e.g. u or r)
    start = text.index(text[-1]) + 1
    end = len(text) - 1
    # Triple quotes
    if text[-3:] in ('"""', "'''"):
        start += 2
        end -= 2
    return text[:start] + 'x' * (end - start) + text[end:]


def parse_udiff(diff, patterns=None, parent='.'):
    """Return a dictionary of matching lines."""
    # For each file of the diff, the entry key is the filename,
    # and the value is a set of row numbers to consider.
    rv = {}
    path = nrows = None
    for line in diff.splitlines():
        if nrows:
            if line[:1] != '-':
                nrows -= 1
            continue
        if line[:3] == '@@ ':
            hunk_match = HUNK_REGEX.match(line)
            (row, nrows) = [int(g or '1') for g in hunk_match.groups()]
            rv[path].update(range(row, row + nrows))
        elif line[:3] == '+++':
            path = line[4:].split('\t', 1)[0]
            if path[:2] == 'b/':
                path = path[2:]
            rv[path] = set()
    return dict([(os.path.join(parent, path), rows)
                 for (path, rows) in rv.items()
                 if rows and filename_match(path, patterns)])


def normalize_paths(value, parent=os.curdir):
    """Parse a comma-separated list of paths.

    Return a list of absolute paths.
    """
    if not value:
        return []
    if isinstance(value, list):
        return value
    paths = []
    for path in value.split(','):
        path = path.strip()
        if '/' in path:
            path = os.path.abspath(os.path.join(parent, path))
        paths.append(path.rstrip('/'))
    return paths


def filename_match(filename, patterns, default=True):
    """Check if patterns contains a pattern that matches filename.

    If patterns is unspecified, this always returns True.
    """
    if not patterns:
        return default
    return any(fnmatch(filename, pattern) for pattern in patterns)


def _is_eol_token(token):
    return token[0] in NEWLINE or token[4][token[3][1]:].lstrip() == '\\\n'
if COMMENT_WITH_NL:
    def _is_eol_token(token, _eol_token=_is_eol_token):
        return _eol_token(token) or (token[0] == tokenize.COMMENT and
                                     token[1] == token[4])

##############################################################################
# Framework to run all checks
##############################################################################


_checks = {'physical_line': {}, 'logical_line': {}, 'tree': {}}


def _get_parameters(function):
    if sys.version_info >= (3, 3):
        return [parameter.name
                for parameter
                in inspect.signature(function).parameters.values()
                if parameter.kind == parameter.POSITIONAL_OR_KEYWORD]
    else:
        return inspect.getargspec(function)[0]


def register_check(check, codes=None):
    """Register a new check object."""
    def _add_check(check, kind, codes, args):
        if check in _checks[kind]:
            _checks[kind][check][0].extend(codes or [])
        else:
            _checks[kind][check] = (codes or [''], args)
    if inspect.isfunction(check):
        args = _get_parameters(check)
        if args and args[0] in ('physical_line', 'logical_line'):
            if codes is None:
                codes = ERRORCODE_REGEX.findall(check.__doc__ or '')
            _add_check(check, args[0], codes, args)
    elif inspect.isclass(check):
        if _get_parameters(check.__init__)[:2] == ['self', 'tree']:
            _add_check(check, 'tree', codes, None)


def init_checks_registry():
    """Register all globally visible functions.

    The first argument name is either 'physical_line' or 'logical_line'.
    """
    mod = inspect.getmodule(register_check)
    for (name, function) in inspect.getmembers(mod, inspect.isfunction):
        register_check(function)
init_checks_registry()


class Checker(object):
    """Load a Python source file, tokenize it, check coding style."""

    def __init__(self, filename=None, lines=None,
                 options=None, report=None, **kwargs):
        if options is None:
            options = StyleGuide(kwargs).options
        else:
            assert not kwargs
        self._io_error = None
        self._physical_checks = options.physical_checks
        self._logical_checks = options.logical_checks
        self._ast_checks = options.ast_checks
        self.max_line_length = options.max_line_length
        self.multiline = False  # in a multiline string?
        self.hang_closing = options.hang_closing
        self.verbose = options.verbose
        self.filename = filename
        # Dictionary where a checker can store its custom state.
        self._checker_states = {}
        if filename is None:
            self.filename = 'stdin'
            self.lines = lines or []
        elif filename == '-':
            self.filename = 'stdin'
            self.lines = stdin_get_value().splitlines(True)
        elif lines is None:
            try:
                self.lines = readlines(filename)
            except IOError:
                (exc_type, exc) = sys.exc_info()[:2]
                self._io_error = '%s: %s' % (exc_type.__name__, exc)
                self.lines = []
        else:
            self.lines = lines
        if self.lines:
            ord0 = ord(self.lines[0][0])
            if ord0 in (0xef, 0xfeff):  # Strip the UTF-8 BOM
                if ord0 == 0xfeff:
                    self.lines[0] = self.lines[0][1:]
                elif self.lines[0][:3] == '\xef\xbb\xbf':
                    self.lines[0] = self.lines[0][3:]
        self.report = report or options.report
        self.report_error = self.report.error

    def report_invalid_syntax(self):
        """Check if the syntax is valid."""
        (exc_type, exc) = sys.exc_info()[:2]
        if len(exc.args) > 1:
            offset = exc.args[1]
            if len(offset) > 2:
                offset = offset[1:3]
        else:
            offset = (1, 0)
        self.report_error(offset[0], offset[1] or 0,
                          'E901 %s: %s' % (exc_type.__name__, exc.args[0]),
                          self.report_invalid_syntax)

    def readline(self):
        """Get the next line from the input buffer."""
        if self.line_number >= self.total_lines:
            return ''
        line = self.lines[self.line_number]
        self.line_number += 1
        if self.indent_char is None and line[:1] in WHITESPACE:
            self.indent_char = line[0]
        return line

    def run_check(self, check, argument_names):
        """Run a check plugin."""
        arguments = []
        for name in argument_names:
            arguments.append(getattr(self, name))
        return check(*arguments)

    def init_checker_state(self, name, argument_names):
        """ Prepares a custom state for the specific checker plugin."""
        if 'checker_state' in argument_names:
            self.checker_state = self._checker_states.setdefault(name, {})

    def check_physical(self, line):
        """Run all physical checks on a raw input line."""
        self.physical_line = line
        for name, check, argument_names in self._physical_checks:
            self.init_checker_state(name, argument_names)
            result = self.run_check(check, argument_names)
            if result is not None:
                (offset, text) = result
                self.report_error(self.line_number, offset, text, check)
                if text[:4] == 'E101':
                    self.indent_char = line[0]

    def build_tokens_line(self):
        """Build a logical line from tokens."""
        logical = []
        comments = []
        length = 0
        prev_row = prev_col = mapping = None
        for token_type, text, start, end, line in self.tokens:
            if token_type in SKIP_TOKENS:
                continue
            if not mapping:
                mapping = [(0, start)]
            if token_type == tokenize.COMMENT:
                comments.append(text)
                continue
            if token_type == tokenize.STRING:
                text = mute_string(text)
            if prev_row:
                (start_row, start_col) = start
                if prev_row != start_row:    # different row
                    prev_text = self.lines[prev_row - 1][prev_col - 1]
                    if prev_text == ',' or (prev_text not in '{[(' and
                                            text not in '}])'):
                        text = ' ' + text
                elif prev_col != start_col:  # different column
                    text = line[prev_col:start_col] + text
            logical.append(text)
            length += len(text)
            mapping.append((length, end))
            (prev_row, prev_col) = end
        self.logical_line = ''.join(logical)
        self.noqa = comments and noqa(''.join(comments))
        return mapping

    def check_logical(self):
        """Build a line from tokens and run all logical checks on it."""
        self.report.increment_logical_line()
        mapping = self.build_tokens_line()

        if not mapping:
            return

        (start_row, start_col) = mapping[0][1]
        start_line = self.lines[start_row - 1]
        self.indent_level = expand_indent(start_line[:start_col])
        if self.blank_before < self.blank_lines:
            self.blank_before = self.blank_lines
        if self.verbose >= 2:
            print(self.logical_line[:80].rstrip())
        for name, check, argument_names in self._logical_checks:
            if self.verbose >= 4:
                print('   ' + name)
            self.init_checker_state(name, argument_names)
            for offset, text in self.run_check(check, argument_names) or ():
                if not isinstance(offset, tuple):
                    for token_offset, pos in mapping:
                        if offset <= token_offset:
                            break
                    offset = (pos[0], pos[1] + offset - token_offset)
                self.report_error(offset[0], offset[1], text, check)
        if self.logical_line:
            self.previous_indent_level = self.indent_level
            self.previous_logical = self.logical_line
        self.blank_lines = 0
        self.tokens = []

    def check_ast(self):
        """Build the file's AST and run all AST checks."""
        try:
            tree = compile(''.join(self.lines), '', 'exec', PyCF_ONLY_AST)
        except (ValueError, SyntaxError, TypeError):
            return self.report_invalid_syntax()
        for name, cls, __ in self._ast_checks:
            checker = cls(tree, self.filename)
            for lineno, offset, text, check in checker.run():
                if not self.lines or not noqa(self.lines[lineno - 1]):
                    self.report_error(lineno, offset, text, check)

    def generate_tokens(self):
        """Tokenize the file, run physical line checks and yield tokens."""
        if self._io_error:
            self.report_error(1, 0, 'E902 %s' % self._io_error, readlines)
        tokengen = tokenize.generate_tokens(self.readline)
        try:
            for token in tokengen:
                if token[2][0] > self.total_lines:
                    return
                self.maybe_check_physical(token)
                yield token
        except (SyntaxError, tokenize.TokenError):
            self.report_invalid_syntax()

    def maybe_check_physical(self, token):
        """If appropriate (based on token), check current physical line(s)."""
        # Called after every token, but act only on end of line.
        if _is_eol_token(token):
            # Obviously, a newline token ends a single physical line.
            self.check_physical(token[4])
        elif token[0] == tokenize.STRING and '\n' in token[1]:
            # Less obviously, a string that contains newlines is a
            # multiline string, either triple-quoted or with internal
            # newlines backslash-escaped. Check every physical line in the
            # string *except* for the last one: its newline is outside of
            # the multiline string, so we consider it a regular physical
            # line, and will check it like any other physical line.
            #
            # Subtleties:
            # - we don't *completely* ignore the last line; if it contains
            #   the magical "# noqa" comment, we disable all physical
            #   checks for the entire multiline string
            # - have to wind self.line_number back because initially it
            #   points to the last line of the string, and we want
            #   check_physical() to give accurate feedback
            if noqa(token[4]):
                return
            self.multiline = True
            self.line_number = token[2][0]
            for line in token[1].split('\n')[:-1]:
                self.check_physical(line + '\n')
                self.line_number += 1
            self.multiline = False

    def check_all(self, expected=None, line_offset=0):
        """Run all checks on the input file."""
        self.report.init_file(self.filename, self.lines, expected, line_offset)
        self.total_lines = len(self.lines)
        if self._ast_checks:
            self.check_ast()
        self.line_number = 0
        self.indent_char = None
        self.indent_level = self.previous_indent_level = 0
        self.previous_logical = ''
        self.tokens = []
        self.blank_lines = self.blank_before = 0
        parens = 0
        for token in self.generate_tokens():
            self.tokens.append(token)
            token_type, text = token[0:2]
            if self.verbose >= 3:
                if token[2][0] == token[3][0]:
                    pos = '[%s:%s]' % (token[2][1] or '', token[3][1])
                else:
                    pos = 'l.%s' % token[3][0]
                print('l.%s\t%s\t%s\t%r' %
                      (token[2][0], pos, tokenize.tok_name[token[0]], text))
            if token_type == tokenize.OP:
                if text in '([{':
                    parens += 1
                elif text in '}])':
                    parens -= 1
            elif not parens:
                if token_type in NEWLINE:
                    if token_type == tokenize.NEWLINE:
                        self.check_logical()
                        self.blank_before = 0
                    elif len(self.tokens) == 1:
                        # The physical line contains only this token.
                        self.blank_lines += 1
                        del self.tokens[0]
                    else:
                        self.check_logical()
                elif COMMENT_WITH_NL and token_type == tokenize.COMMENT:
                    if len(self.tokens) == 1:
                        # The comment also ends a physical line
                        token = list(token)
                        token[1] = text.rstrip('\r\n')
                        token[3] = (token[2][0], token[2][1] + len(token[1]))
                        self.tokens = [tuple(token)]
                        self.check_logical()
        if self.tokens:
            self.check_physical(self.lines[-1])
            self.check_logical()
        return self.report.get_file_results()


class BaseReport(object):
    """Collect the results of the checks."""

    print_filename = False

    def __init__(self, options):
        self._benchmark_keys = options.benchmark_keys
        self._ignore_code = options.ignore_code
        # Results
        self.elapsed = 0
        self.total_errors = 0
        self.counters = dict.fromkeys(self._benchmark_keys, 0)
        self.messages = {}

    def start(self):
        """Start the timer."""
        self._start_time = time.time()

    def stop(self):
        """Stop the timer."""
        self.elapsed = time.time() - self._start_time

    def init_file(self, filename, lines, expected, line_offset):
        """Signal a new file."""
        self.filename = filename
        self.lines = lines
        self.expected = expected or ()
        self.line_offset = line_offset
        self.file_errors = 0
        self.counters['files'] += 1
        self.counters['physical lines'] += len(lines)

    def increment_logical_line(self):
        """Signal a new logical line."""
        self.counters['logical lines'] += 1

    def error(self, line_number, offset, text, check):
        """Report an error, according to options."""
        code = text[:4]
        if self._ignore_code(code):
            return
        if code in self.counters:
            self.counters[code] += 1
        else:
            self.counters[code] = 1
            self.messages[code] = text[5:]
        # Don't care about expected errors or warnings
        if code in self.expected:
            return
        if self.print_filename and not self.file_errors:
            print(self.filename)
        self.file_errors += 1
        self.total_errors += 1
        return code

    def get_file_results(self):
        """Return the count of errors and warnings for this file."""
        return self.file_errors

    def get_count(self, prefix=''):
        """Return the total count of errors and warnings."""
        return sum([self.counters[key]
                    for key in self.messages if key.startswith(prefix)])

    def get_statistics(self, prefix=''):
        """Get statistics for message codes that start with the prefix.

        prefix='' matches all errors and warnings
        prefix='E' matches all errors
        prefix='W' matches all warnings
        prefix='E4' matches all errors that have to do with imports
        """
        return ['%-7s %s %s' % (self.counters[key], key, self.messages[key])
                for key in sorted(self.messages) if key.startswith(prefix)]

    def print_statistics(self, prefix=''):
        """Print overall statistics (number of errors and warnings)."""
        for line in self.get_statistics(prefix):
            print(line)

    def print_benchmark(self):
        """Print benchmark numbers."""
        print('%-7.2f %s' % (self.elapsed, 'seconds elapsed'))
        if self.elapsed:
            for key in self._benchmark_keys:
                print('%-7d %s per second (%d total)' %
                      (self.counters[key] / self.elapsed, key,
                       self.counters[key]))


class FileReport(BaseReport):
    """Collect the results of the checks and print only the filenames."""
    print_filename = True


class StandardReport(BaseReport):
    """Collect and print the results of the checks."""

    def __init__(self, options):
        super(StandardReport, self).__init__(options)
        self._fmt = REPORT_FORMAT.get(options.format.lower(),
                                      options.format)
        self._repeat = options.repeat
        self._show_source = options.show_source
        self._show_pep8 = options.show_pep8

    def init_file(self, filename, lines, expected, line_offset):
        """Signal a new file."""
        self._deferred_print = []
        return super(StandardReport, self).init_file(
            filename, lines, expected, line_offset)

    def error(self, line_number, offset, text, check):
        """Report an error, according to options."""
        code = super(StandardReport, self).error(line_number, offset,
                                                 text, check)
        if code and (self.counters[code] == 1 or self._repeat):
            self._deferred_print.append(
                (line_number, offset, code, text[5:], check.__doc__))
        return code

    def get_file_results(self):
        """Print the result and return the overall count for this file."""
        self._deferred_print.sort()
        for line_number, offset, code, text, doc in self._deferred_print:
            print(self._fmt % {
                'path': self.filename,
                'row': self.line_offset + line_number, 'col': offset + 1,
                'code': code, 'text': text,
            })
            if self._show_source:
                if line_number > len(self.lines):
                    line = ''
                else:
                    line = self.lines[line_number - 1]
                print(line.rstrip())
                print(re.sub(r'\S', ' ', line[:offset]) + '^')
            if self._show_pep8 and doc:
                print('    ' + doc.strip())

            # stdout is block buffered when not stdout.isatty().
            # line can be broken where buffer boundary since other processes
            # write to same file.
            # flush() after print() to avoid buffer boundary.
            # Typical buffer size is 8192. line written safely when
            # len(line) < 8192.
            sys.stdout.flush()
        return self.file_errors


class DiffReport(StandardReport):
    """Collect and print the results for the changed lines only."""

    def __init__(self, options):
        super(DiffReport, self).__init__(options)
        self._selected = options.selected_lines

    def error(self, line_number, offset, text, check):
        if line_number not in self._selected[self.filename]:
            return
        return super(DiffReport, self).error(line_number, offset, text, check)


class StyleGuide(object):
    """Initialize a PEP-8 instance with few options."""

    def __init__(self, *args, **kwargs):
        # build options from the command line
        self.checker_class = kwargs.pop('checker_class', Checker)
        parse_argv = kwargs.pop('parse_argv', False)
        config_file = kwargs.pop('config_file', False)
        parser = kwargs.pop('parser', None)
        # build options from dict
        options_dict = dict(*args, **kwargs)
        arglist = None if parse_argv else options_dict.get('paths', None)
        options, self.paths = process_options(
            arglist, parse_argv, config_file, parser)
        if options_dict:
            options.__dict__.update(options_dict)
            if 'paths' in options_dict:
                self.paths = options_dict['paths']

        self.runner = self.input_file
        self.options = options

        if not options.reporter:
            options.reporter = BaseReport if options.quiet else StandardReport

        options.select = tuple(options.select or ())
        if not (options.select or options.ignore or
                options.testsuite or options.doctest) and DEFAULT_IGNORE:
            # The default choice: ignore controversial checks
            options.ignore = tuple(DEFAULT_IGNORE.split(','))
        else:
            # Ignore all checks which are not explicitly selected
            options.ignore = ('',) if options.select else tuple(options.ignore)
        options.benchmark_keys = BENCHMARK_KEYS[:]
        options.ignore_code = self.ignore_code
        options.physical_checks = self.get_checks('physical_line')
        options.logical_checks = self.get_checks('logical_line')
        options.ast_checks = self.get_checks('tree')
        self.init_report()

    def init_report(self, reporter=None):
        """Initialize the report instance."""
        self.options.report = (reporter or self.options.reporter)(self.options)
        return self.options.report

    def check_files(self, paths=None):
        """Run all checks on the paths."""
        if paths is None:
            paths = self.paths
        report = self.options.report
        runner = self.runner
        report.start()
        try:
            for path in paths:
                if os.path.isdir(path):
                    self.input_dir(path)
                elif not self.excluded(path):
                    runner(path)
        except KeyboardInterrupt:
            print('... stopped')
        report.stop()
        return report

    def input_file(self, filename, lines=None, expected=None, line_offset=0):
        """Run all checks on a Python source file."""
        if self.options.verbose:
            print('checking %s' % filename)
        fchecker = self.checker_class(
            filename, lines=lines, options=self.options)
        return fchecker.check_all(expected=expected, line_offset=line_offset)

    def input_dir(self, dirname):
        """Check all files in this directory and all subdirectories."""
        dirname = dirname.rstrip('/')
        if self.excluded(dirname):
            return 0
        counters = self.options.report.counters
        verbose = self.options.verbose
        filepatterns = self.options.filename
        runner = self.runner
        for root, dirs, files in os.walk(dirname):
            if verbose:
                print('directory ' + root)
            counters['directories'] += 1
            for subdir in sorted(dirs):
                if self.excluded(subdir, root):
                    dirs.remove(subdir)
            for filename in sorted(files):
                # contain a pattern that matches?
                if ((filename_match(filename, filepatterns) and
                     not self.excluded(filename, root))):
                    runner(os.path.join(root, filename))

    def excluded(self, filename, parent=None):
        """Check if the file should be excluded.

        Check if 'options.exclude' contains a pattern that matches filename.
        """
        if not self.options.exclude:
            return False
        basename = os.path.basename(filename)
        if filename_match(basename, self.options.exclude):
            return True
        if parent:
            filename = os.path.join(parent, filename)
        filename = os.path.abspath(filename)
        return filename_match(filename, self.options.exclude)

    def ignore_code(self, code):
        """Check if the error code should be ignored.

        If 'options.select' contains a prefix of the error code,
        return False.  Else, if 'options.ignore' contains a prefix of
        the error code, return True.
        """
        if len(code) < 4 and any(s.startswith(code)
                                 for s in self.options.select):
            return False
        return (code.startswith(self.options.ignore) and
                not code.startswith(self.options.select))

    def get_checks(self, argument_name):
        """Get all the checks for this category.

        Find all globally visible functions where the first argument name
        starts with argument_name and which contain selected tests.
        """
        checks = []
        for check, attrs in _checks[argument_name].items():
            (codes, args) = attrs
            if any(not (code and self.ignore_code(code)) for code in codes):
                checks.append((check.__name__, check, args))
        return sorted(checks)


def get_parser(prog='pep8', version=__version__):
    parser = OptionParser(prog=prog, version=version,
                          usage="%prog [options] input ...")
    parser.config_options = [
        'exclude', 'filename', 'select', 'ignore', 'max-line-length',
        'hang-closing', 'count', 'format', 'quiet', 'show-pep8',
        'show-source', 'statistics', 'verbose']
    parser.add_option('-v', '--verbose', default=0, action='count',
                      help="print status messages, or debug with -vv")
    parser.add_option('-q', '--quiet', default=0, action='count',
                      help="report only file names, or nothing with -qq")
    parser.add_option('-r', '--repeat', default=True, action='store_true',
                      help="(obsolete) show all occurrences of the same error")
    parser.add_option('--first', action='store_false', dest='repeat',
                      help="show first occurrence of each error")
    parser.add_option('--exclude', metavar='patterns', default=DEFAULT_EXCLUDE,
                      help="exclude files or directories which match these "
                           "comma separated patterns (default: %default)")
    parser.add_option('--filename', metavar='patterns', default='*.py',
                      help="when parsing directories, only check filenames "
                           "matching these comma separated patterns "
                           "(default: %default)")
    parser.add_option('--select', metavar='errors', default='',
                      help="select errors and warnings (e.g. E,W6)")
    parser.add_option('--ignore', metavar='errors', default='',
                      help="skip errors and warnings (e.g. E4,W) "
                           "(default: %s)" % DEFAULT_IGNORE)
    parser.add_option('--show-source', action='store_true',
                      help="show source code for each error")
    parser.add_option('--show-pep8', action='store_true',
                      help="show text of PEP 8 for each error "
                           "(implies --first)")
    parser.add_option('--statistics', action='store_true',
                      help="count errors and warnings")
    parser.add_option('--count', action='store_true',
                      help="print total number of errors and warnings "
                           "to standard error and set exit code to 1 if "
                           "total is not null")
    parser.add_option('--max-line-length', type='int', metavar='n',
                      default=MAX_LINE_LENGTH,
                      help="set maximum allowed line length "
                           "(default: %default)")
    parser.add_option('--hang-closing', action='store_true',
                      help="hang closing bracket instead of matching "
                           "indentation of opening bracket's line")
    parser.add_option('--format', metavar='format', default='default',
                      help="set the error format [default|pylint|<custom>]")
    parser.add_option('--diff', action='store_true',
                      help="report changes only within line number ranges in "
                           "the unified diff received on STDIN")
    group = parser.add_option_group("Testing Options")
    if os.path.exists(TESTSUITE_PATH):
        group.add_option('--testsuite', metavar='dir',
                         help="run regression tests from dir")
        group.add_option('--doctest', action='store_true',
                         help="run doctest on myself")
    group.add_option('--benchmark', action='store_true',
                     help="measure processing speed")
    return parser


def read_config(options, args, arglist, parser):
    """Read and parse configurations

    If a config file is specified on the command line with the "--config"
    option, then only it is used for configuration.

    Otherwise, the user configuration (~/.config/pep8) and any local
    configurations in the current directory or above will be merged together
    (in that order) using the read method of ConfigParser.
    """
    config = RawConfigParser()

    cli_conf = options.config

    local_dir = os.curdir

    if USER_CONFIG and os.path.isfile(USER_CONFIG):
        if options.verbose:
            print('user configuration: %s' % USER_CONFIG)
        config.read(USER_CONFIG)

    parent = tail = args and os.path.abspath(os.path.commonprefix(args))
    while tail:
        if config.read(os.path.join(parent, fn) for fn in PROJECT_CONFIG):
            local_dir = parent
            if options.verbose:
                print('local configuration: in %s' % parent)
            break
        (parent, tail) = os.path.split(parent)

    if cli_conf and os.path.isfile(cli_conf):
        if options.verbose:
            print('cli configuration: %s' % cli_conf)
        config.read(cli_conf)

    pep8_section = parser.prog
    if config.has_section(pep8_section):
        option_list = dict([(o.dest, o.type or o.action)
                            for o in parser.option_list])

        # First, read the default values
        (new_options, __) = parser.parse_args([])

        # Second, parse the configuration
        for opt in config.options(pep8_section):
            if opt.replace('_', '-') not in parser.config_options:
                print("  unknown option '%s' ignored" % opt)
                continue
            if options.verbose > 1:
                print("  %s = %s" % (opt, config.get(pep8_section, opt)))
            normalized_opt = opt.replace('-', '_')
            opt_type = option_list[normalized_opt]
            if opt_type in ('int', 'count'):
                value = config.getint(pep8_section, opt)
            elif opt_type == 'string':
                value = config.get(pep8_section, opt)
                if normalized_opt == 'exclude':
                    value = normalize_paths(value, local_dir)
            else:
                assert opt_type in ('store_true', 'store_false')
                value = config.getboolean(pep8_section, opt)
            setattr(new_options, normalized_opt, value)

        # Third, overwrite with the command-line options
        (options, __) = parser.parse_args(arglist, values=new_options)
    options.doctest = options.testsuite = False
    return options


def process_options(arglist=None, parse_argv=False, config_file=None,
                    parser=None):
    """Process options passed either via arglist or via command line args.

    Passing in the ``config_file`` parameter allows other tools, such as flake8
    to specify their own options to be processed in pep8.
    """
    if not parser:
        parser = get_parser()
    if not parser.has_option('--config'):
        group = parser.add_option_group("Configuration", description=(
            "The project options are read from the [%s] section of the "
            "tox.ini file or the setup.cfg file located in any parent folder "
            "of the path(s) being processed.  Allowed options are: %s." %
            (parser.prog, ', '.join(parser.config_options))))
        group.add_option('--config', metavar='path', default=config_file,
                         help="user config file location")
    # Don't read the command line if the module is used as a library.
    if not arglist and not parse_argv:
        arglist = []
    # If parse_argv is True and arglist is None, arguments are
    # parsed from the command line (sys.argv)
    (options, args) = parser.parse_args(arglist)
    options.reporter = None

    if options.ensure_value('testsuite', False):
        args.append(options.testsuite)
    elif not options.ensure_value('doctest', False):
        if parse_argv and not args:
            if options.diff or any(os.path.exists(name)
                                   for name in PROJECT_CONFIG):
                args = ['.']
            else:
                parser.error('input not specified')
        options = read_config(options, args, arglist, parser)
        options.reporter = parse_argv and options.quiet == 1 and FileReport

    options.filename = _parse_multi_options(options.filename)
    options.exclude = normalize_paths(options.exclude)
    options.select = _parse_multi_options(options.select)
    options.ignore = _parse_multi_options(options.ignore)

    if options.diff:
        options.reporter = DiffReport
        stdin = stdin_get_value()
        options.selected_lines = parse_udiff(stdin, options.filename, args[0])
        args = sorted(options.selected_lines)

    return options, args


def _parse_multi_options(options, split_token=','):
    r"""Split and strip and discard empties.

    Turns the following:

    A,
    B,

    into ["A", "B"]
    """
    if options:
        return [o.strip() for o in options.split(split_token) if o.strip()]
    else:
        return options


def _main():
    """Parse options and run checks on Python source."""
    import signal

    # Handle "Broken pipe" gracefully
    try:
        signal.signal(signal.SIGPIPE, lambda signum, frame: sys.exit(1))
    except AttributeError:
        pass    # not supported on Windows

    pep8style = StyleGuide(parse_argv=True)
    options = pep8style.options

    if options.doctest or options.testsuite:
        from testsuite.support import run_tests
        report = run_tests(pep8style)
    else:
        report = pep8style.check_files()

    if options.statistics:
        report.print_statistics()

    if options.benchmark:
        report.print_benchmark()

    if options.testsuite and not options.quiet:
        report.print_results()

    if report.total_errors:
        if options.count:
            sys.stderr.write(str(report.total_errors) + '\n')
        sys.exit(1)

if __name__ == '__main__':
    _main()
