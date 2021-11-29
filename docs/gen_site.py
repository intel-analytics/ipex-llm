#!/usr/bin/env python

#
# Copyright 2018 Analytics Zoo Authors.
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
# Usage ###################
# Run ./gen_site.py to build site with Analytics Zoo docs with following commands
# -s: add scala docs
# -p: add python docs
# -m [port]: --startserver
# -h: help
# Example
# ./gen_site.py -s -p -m 8080
############################

import argparse
import sys
import os
import subprocess


parser = argparse.ArgumentParser(description='Process Analytics Zoo docs.')
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
                    help='Use local zoo doc repo(if it exists) instead of downloading from remote')

args = parser.parse_args()

scaladocs = args.scaladocsflag

pythondocs = args.pythondocsflag

local_doc = args.local_doc

script_path = os.path.realpath(__file__)
dir_name = os.path.dirname(script_path)
os.chdir(dir_name)

# check if mkdoc is installed
subprocess.run(['mkdocs', '--version'])  # pip install mkdocs==0.16.3

# refresh local docs repo
if not (local_doc and os.path.isdir("/tmp/zoo-doc")):
    subprocess.run(['rm', '-rf', '/tmp/zoo-doc'])  # rm doc repo
    # git clone doc repo
    subprocess.run(['git', 'clone', 'https://github.com/analytics-zoo/analytics-zoo.github.io.git', '/tmp/zoo-doc'])

# refresh theme folder
subprocess.run(['rm', '-rf', '{}/mkdocs_windmill'.format(dir_name)])  # rm theme folder
subprocess.run(['cp', '-r', '/tmp/zoo-doc/mkdocs_windmill', dir_name])

# refresh css file
subprocess.run(['cp', '/tmp/zoo-doc/extra.css', '{}/docs'.format(dir_name)])  # mv theme folder

# mkdocs build
subprocess.run(['mkdocs', 'build'])

# replace resources folder in site
# mv theme folder
subprocess.run(' '.join(['cp', '/tmp/zoo-doc/css/*', '{}/site/css'.format(dir_name)]), shell=True)
subprocess.run(' '.join(['cp', '/tmp/zoo-doc/js/*', '{}/site/js'.format(dir_name)]), shell=True)
subprocess.run(' '.join(['cp', '/tmp/zoo-doc/fonts/*', '{}/site/fonts'.format(dir_name)]), shell=True)
subprocess.run(' '.join(['cp', '/tmp/zoo-doc/img/*', '{}/site/img'.format(dir_name)]), shell=True)
subprocess.run(' '.join(['cp', '/tmp/zoo-doc/version-list', '{}/site'.format(dir_name)]), shell=True)

if scaladocs:
    print('build scala doc')
    zoo_dir = os.path.dirname(dir_name)
    os.chdir(zoo_dir)
    subprocess.run(['mvn', 'scala:doc'])  # build scala doc
    scaladocs_dir = zoo_dir + '/zoo/target/site/scaladocs/'
    target_dir = dir_name + '/site/APIGuide/'
    if not os.path.exists(target_dir):
        subprocess.run(['mkdir', target_dir])  # mkdir APIGuide
    # mv scaladocs
    subprocess.run(' '.join(['cp', '-r', scaladocs_dir, target_dir + 'scaladoc/']), shell=True)

if pythondocs:
    print('build python')
    pyspark_dir = os.path.dirname(dir_name) + '/pyzoo/docs/'
    target_dir = dir_name + '/site/APIGuide/'
    os.chdir(pyspark_dir)
    subprocess.run(['./doc-gen.sh'])  # build python doc
    pythondocs_dir = pyspark_dir + '_build/html/'
    if not os.path.exists(target_dir):
        subprocess.run(['mkdir', target_dir])  # mkdir APIGuide
    # mv pythondocs
    subprocess.run(' '.join(['cp', '-r', pythondocs_dir, target_dir + 'python-api-doc/']), shell=True)

os.chdir(dir_name)

if args.debugport:
    print('starting mkdoc server in debug mode')
    addr = '--dev-addr=*:'+str(args.debugport)
    # mkdocs start serve
    subprocess.run(['mkdocs', 'serve', addr])

if args.port:
    os.chdir(dir_name + '/site')
    # start http server
    subprocess.run(['python', '-m', 'http.server', '{}'.format(args.port)])
