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
