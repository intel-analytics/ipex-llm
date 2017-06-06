import re

def _process_docstring(app, what, name, obj, options, lines):
    #quote_re = re.compile(r'\s*["\']{3}(?:.(?<!["\']{3}))*$')
    param_re = re.compile(r'\s*:.*$')
    blank_re = re.compile(r'\s*$')
    prmpt_re = re.compile(r'\s*>>>.*$')
    liter_re = re.compile(r'\s*```\s*$')

    #quote_flag = False
    param_flag = False
    blank_flag = False
    prmpt_flag = False
    liter_flag = False

    paraform_re = re.compile(r'(^\s*:param\s+)(\S+)(?<!:)(\s+.*$)')
    at_re = re.compile(r'(^\s*)@')

    offset = 0
    for j in range(len(lines)):
        i = j+offset
        line = lines[i]
        line = line+' '
        line = paraform_re.sub(r'\1\2:\3', line)
        line = line[:-1]
        line = at_re.sub(r'\1:', line)
        #line = line.lstrip()
        #line = re.sub('^\s{4}','',line)
        line = '\n' if line == '' else line
        lines[i] = line
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
            lines[i] = line.lstrip()
            lines.insert(i,'')
            offset += 1
        # normal param line
        elif param_re.match(line):
            param_flag = True
            blank_flag = False
            prmpt_flag = False
            lines[i] = line.lstrip()
        # fisrt prompt line with no head blank line 
        elif not prmpt_flag and not blank_flag and prmpt_re.match(line):
            param_falg = False
            blank_flag = False
            prmpt_flag = True
            lines[i]=line.lstrip()
            lines.insert(i, '')
            offset += 1
        # normal param line
        elif prmpt_re.match(line):
            param_flag = False
            blank_flag = False
            prmpt_flag = True
            lines[i]=line.lstrip()
        # first literal block line
        elif not liter_flag and liter_re.match(line):
            param_flag = False
            blank_flag = False
            prmpt_flag = False
            liter_flag = True
            lines[i]='::'
        # last literal block line
        elif liter_flag and liter_re.match(line):
            param_flag = False
            blank_flag = True
            prmpt_flag = False
            liter_flag = False
            lines[i]=''
        # regular line with head prmpt line
        elif prmpt_flag: 
            param_flag = False
            blank_flag = False
            prmpt_flag = True
            lines[i]=line.lstrip()
        # regular line with head param line
        elif param_flag:
            param_flag = True
            blank_flag = False
            prmpt_flag = False
            lines[i-1] += line.lstrip()
            lines[i:-1] = lines[i+1:]
            lines.pop()
            offset -= 1
        # regular line within literal block
        elif liter_flag:
            line = ' '+line
            lines[i]=line
        # regualr line
        else:
            param_flag = False
            blank_flag = False
            prmpt_flag = False
            lines[i]=line.lstrip()
            #lines[i] = '|' + line

def setup(app):
    app.connect("autodoc-process-docstring", _process_docstring)
