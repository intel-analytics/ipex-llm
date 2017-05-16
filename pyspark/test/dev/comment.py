#!/usr/bin/env python

#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import re
import os

def _process_docstring(filepath, lines):
    quote_re = re.compile(r'\s*["\']{3}(?:.(?<!["\']{3}))*$')
    param_re = re.compile(r'^\s*:?@?\s*(?:param\s+|return:?).*$')
    blank_re = re.compile(r'\s*$')
    prmpt_re = re.compile(r'\s*>>>.*$')

    quote_flag = False
    param_flag = False
    blank_flag = False
    prmpt_flag = False

    param_rules = [(re.compile(x), y) for x,y in [
            ('^\s*.{0,2}(?<!:)(?:param\s+|return:?).*$', 'A colon is required to be attached just before \'param\', and no blank gap is allowed. For example, \':param\', not \': param\' or \'@param\'. For \'return\', a colon is also required, like \':return\'.' ),
            ('^\s*:(?:param\s+\S+(?<!:)|return)\s+?.*$', 'A colon is required to be attached just after \':param xx\', and no blank gap is allowed. For example, \':param xx:\', not \':param xx :\' or \':param xx\'. For \'return\', a colon is also required, lik \':return:\'.')]]

    for i in range(len(lines)):
        line = lines[i][:-1]
        ln = i+1
        if not quote_flag and quote_re.match(line):
            quote_flag = True
            blank_flag = False
            param_flag = False
            prmpt_flag = False
        elif quote_flag and quote_re.match(line):
            quote_flag = False
            param_flag = False
            blank_flag = False
            prmpt_flag = False
        elif quote_flag:
            for param_rule in param_rules:
                rule_re, err_mes = param_rule
                if rule_re.match(line):
                    raise Exception('line {0}, in {3}:\n {1}\n{2}'.format(ln, line, err_mes, filepath))
            # blank line
            if blank_re.match(line):
                param_flag = False
                blank_flag = True
                prmpt_flag = False
            # first param line with no head blank line
            elif not param_flag and not blank_flag and param_re.match(line):
                param_flag = True
                blank_flag = False
                prmpt_falg = False
                err_mes = 'A blank line preceding the first parameter line is required. Try adding a blank line before it.'
                raise Exception('line {0}, in {3}:\n {1}\n{2}'.format(ln, line , err_mes, filepath))
            # normal param line
            elif param_re.match(line):
                param_flag = True
                blank_flag = False
                prmpt_flag = False
            # fisrt prompt line with no head blank line 
            elif not prmpt_flag and not blank_flag and prmpt_re.match(line):
                param_falg = False
                blank_flag = False
                prmpt_flag = True
                err_mes = 'A blank line preceding the first prompt line is required. Try adding a blank line before it.'
                raise Exception('line {0}, in {3}:\n {1}\n{2}'.format(ln, line , err_mes, filepath))
            # normal param line
            elif prmpt_re.match(line):
                param_flag = False
                blank_flag = False
                prmpt_flag = True
            # regular line with head prmpt line
            elif prmpt_flag: 
                param_flag = False
                blank_flag = False
                prmpt_flag = True
            # regular line with head param line
            elif param_flag:
                param_flag = True
                blank_flag = False
                prmpt_flag = False
                err_mes = 'Add a blank line before it to make it as a new paragraph. Or append it to the previous parameter line. You have to write the parameter definition in one line.'
                raise Exception('line {0}, in {3}:\n {1}\n{2}'.format(ln, line , err_mes, filepath))
            # regualr line
            else:
                param_flag = False
                blank_flag = False
                prmpt_flag = False

if __name__ == '__main__':
    python_nn_root = "./pyspark/dl/"
    for dirpath, dirnames, filenames in os.walk(python_nn_root):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            _process_docstring(filepath, open(filepath).readlines())

