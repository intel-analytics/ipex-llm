#!/usr/bin/env python
# 
# All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
Form a subset of the Flickr Style data, download images store for BigDL training file.
"""
import commands
import os
import urllib
import hashlib
import argparse
import numpy as np
import pandas as pd
from skimage import io
import multiprocessing

# Flickr returns a special image if the request is unavailable.
MISSING_IMAGE_SHA1 = '6a92790b1c2a301c6e7ddef645dca1f53ea97ac2'

example_dirname = os.path.abspath(os.path.dirname(__file__))

def download_image(args_tuple):
    try:
        url, filename = args_tuple
        if not os.path.exists(filename):
            urllib.urlretrieve(url, filename)
        with open(filename) as f:
            assert hashlib.sha1(f.read()).hexdigest() != MISSING_IMAGE_SHA1
        return True
    except KeyboardInterrupt:
        raise Exception()
    except:
        return False

def partition_with_file(label_file, type):
    parent_dir = os.path.dirname(label_file)
    file = open(label_file)
    # read all lines
    lines = list(file)
    for line in lines:
        file = line.split("\n")[0].split(" ")[0]
        label = line.split("\n")[0].split(" ")[1]
        dirname = parent_dir + '/' + type + '/' + label
        if not file_exists(dirname):
            commands.getoutput("mkdir -p %s" % (dirname))
        commands.getoutput('cp %s %s/' % (file, dirname))


def file_exists(file_path):
    if "No such file or directory" in commands.getoutput('ls %s' % file_path):
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download a subset of Flickr Style to a directory')
    parser.add_argument(
        '-d', '--dest', type=str, default=".",
        help="destination directory")
    parser.add_argument(
        '-i', '--images', type=int, default=-1,
        help="number of images to download (-1 for all [default])",
    )

    args = parser.parse_args()
    np.random.seed(12345)

    # Read image url information.
    csv_filename = os.path.join(example_dirname, 'flickr_style.csv.gz')
    df = pd.read_csv(csv_filename, index_col=0, compression='gzip')
    df = df.iloc[np.random.permutation(df.shape[0])]
    if args.images > 0 and args.images < df.shape[0]:
        df = df.iloc[:args.images]

    # Create image folder
    dest = args.dest
    images_dirname = os.path.join(dest, 'images')
    if not os.path.exists(images_dirname):
        os.makedirs(images_dirname)
    df['image_filename'] = [
        os.path.join(images_dirname, _.split('/')[-1]) for _ in df['image_url']
        ]

    # Download images.
    num_workers = multiprocessing.cpu_count() - 1
    print('Downloading {} images ...'.format(
        df.shape[0]))
    pool = multiprocessing.Pool(processes=num_workers)
    map_args = zip(df['image_url'], df['image_filename'])
    results = pool.map(download_image, map_args)

    # write out training/test files.
    df = df[results]
    for split in ['train', 'test']:
        split_df = df[df['_split'] == split]
        filename = os.path.join(dest, '{}.txt'.format(split))
        split_df[['image_filename', 'label']].to_csv(
            filename, sep=' ', header=None, index=None)
        partition_with_file(filename, split)
        commands.getoutput('rm %s' % (filename))
    print('Stored train/test images.'.format(
        df.shape[0]))

    # Remove image folder
    commands.getoutput('rm -r %s' % (images_dirname))
