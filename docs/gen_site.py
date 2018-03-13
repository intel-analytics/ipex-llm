#!/usr/bin/env python

import argparse
import sys
import os
from subprocess import Popen, PIPE
import subprocess


def run_cmd(cmds, err_msg, s=False):
    cmd = cmds
    if s:
        cmd = ' '.join(cmds)
    try:
        # print cmd
        p = Popen(cmd, shell=s)
        p.communicate()
        if p.returncode != 0:
            print err_msg
            sys.exit(1)
    except OSError as e:
        print err_msg
        print e.strerror
        sys.exit(1)

parser = argparse.ArgumentParser(description='Process BigDL docs.')
parser.add_argument('-s', '--scaladocs',
    dest='scaladocsflag', action='store_true',
    help='Add scala doc to site')
parser.add_argument('-p', '--pythondocs',
    dest='pythondocsflag', action='store_true',
    help='Add python doc to site')
parser.add_argument('-m', '--startserver',
    dest='port', type=int,
    help='Start server at PORT after building')
parser.add_argument('-d', '--startmkdocserve',
    dest='debugport', type=int,
    help=argparse.SUPPRESS)
parser.add_argument('-l', '--localdoc',
    dest='local_doc', action='store_true',
    help='Use local bigdl doc repo(if it exists) instead of downloading from remote')

args = parser.parse_args()

scaladocs = args.scaladocsflag

pythondocs = args.pythondocsflag

local_doc = args.local_doc

script_path = os.path.realpath(__file__)
dir_name = os.path.dirname(script_path)
os.chdir(dir_name)

# check if mkdoc is installed
run_cmd(['mkdocs', '--version'],
    'Please install mkdocs and run this script again\n\te.g., pip install mkdocs')

# refresh local docs repo
if not (local_doc and os.path.isdir("/tmp/bigdl-doc")):
    run_cmd(['rm', '-rf', '/tmp/bigdl-doc'],
        'rm doc repo error')
    run_cmd(['git', 'clone', 'https://github.com/bigdl-project/bigdl-project.github.io.git', '/tmp/bigdl-doc'],
        'git clone doc repo error')

# refresh theme folder
run_cmd(['rm', '-rf', '{}/mkdocs_windmill'.format(dir_name)],
    'rm theme folder error')
run_cmd(['cp', '-r', '/tmp/bigdl-doc/mkdocs_windmill', dir_name],
    'mv theme foler error')

# refresh css file
run_cmd(['cp', '/tmp/bigdl-doc/extra.css', '{}/docs'.format(dir_name)],
    'mv theme foler error')

# mkdocs build
run_cmd(['mkdocs', 'build'],
    'mkdocs build error')

# replace resources folder in site
run_cmd(['cp', '/tmp/bigdl-doc/css/*', '{}/site/css'.format(dir_name)],
    'mv theme foler error', s=True)
run_cmd(['cp', '/tmp/bigdl-doc/js/*', '{}/site/js'.format(dir_name)],
    'mv theme foler error', s=True)
run_cmd(['cp', '/tmp/bigdl-doc/fonts/*', '{}/site/fonts'.format(dir_name)],
    'mv theme foler error', s=True)
run_cmd(['cp', '/tmp/bigdl-doc/img/*', '{}/site/img'.format(dir_name)],
    'mv theme foler error', s=True)
run_cmd(['cp', '/tmp/bigdl-doc/version-list', '{}/site'.format(dir_name)],
    'mv theme foler error', s=True)

if scaladocs:
    print 'build scala doc'
    bigdl_dir = os.path.dirname(dir_name)
    os.chdir(bigdl_dir)
    run_cmd(['mvn', 'scala:doc'], 'Build scala doc error')
    scaladocs_dir = bigdl_dir + '/spark/dl/target/site/scaladocs/*'
    target_dir = dir_name + '/site/APIGuide/scaladoc/'
    run_cmd(['cp', '-r', scaladocs_dir, target_dir],
        'mv scaladocs error', s=True)

if pythondocs:
    print 'build python'
    pyspark_dir = os.path.dirname(dir_name) + '/pyspark/docs/'
    target_dir = dir_name + '/site/APIGuide/python-api-doc/'
    os.chdir(pyspark_dir)
    run_cmd(['./doc-gen.sh'], 'Build python doc error')
    pythondocs_dir = pyspark_dir + '_build/html/*'
    run_cmd(['cp', '-r', pythondocs_dir, target_dir],
        'mv scaladocs error', s=True)

os.chdir(dir_name)

if args.debugport != None:
    print 'starting mkdoc server in debug mode'
    addr = '--dev-addr=*:'+str(args.debugport)
    run_cmd(['mkdocs', 'serve', addr],
         'mkdocs start serve error')

if args.port != None:
    os.chdir(dir_name + '/site')
    run_cmd(['python', '-m', 'SimpleHTTPServer', '{}'.format(args.port)],
        'start http server error')

