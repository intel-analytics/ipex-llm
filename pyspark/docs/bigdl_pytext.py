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

def _process_docstring(app, what, name, obj, options, lines):
    liter_re = re.compile(r'\s*```\s*$')

    liter_flag = False

    offset = 0
    for j in range(len(lines)):
        i = j+offset
        line = lines[i]
        # first literal block line
        if not liter_flag and liter_re.match(line):
            liter_flag = True
            lines.insert(i+1,'')
            offset += 1
            lines[i]='::'
        # last literal block line
        elif liter_flag and liter_re.match(line):
            liter_flag = False
            lines[i]=''
        # regular line within literal block
        elif liter_flag:
            line = ' '+line
            lines[i]=line
        # regualr line
        else:
            lines[i]=line.lstrip()

def setup(app):
    app.connect("autodoc-process-docstring", _process_docstring)
