import sys
import os

script_path = os.path.realpath(__file__)
dir_name = os.path.dirname(script_path)
doc_dir = dir_name + '/docs/APIdocs'

def clean_merged():
    for root, dirs, files in os.walk(doc_dir):
        merged_filename = '_merged_' + root[root.rfind('/')+1:] + '.md'
        if merged_filename in files:
            os.remove(os.path.join(root, merged_filename))

def merge_mds():
    for root, dirs, files in os.walk(doc_dir):
        merged_filename = '_merged_' + root[root.rfind('/')+1:] + '.md'
        #remove merged md if exists
        if merged_filename in files:
            os.remove(os.path.join(root, merged_filename))
        #only merge normal md's
        md_files = filter(lambda x: os.path.splitext(x)[1] == '.md' and not '_merged_' in x, files)
        #if no md's, skip this folder
        if len(md_files) == 0:
            continue
        print "merging md files in dir",root
        with open(os.path.join(root, merged_filename), 'w') as f:
            print "-->files found to be merged: ",','.join(md_files)
            for md in md_files:
                with open(os.path.join(root, md), 'r') as f_md:
                    for line in f_md:
                        f.write(line)
                #write a blank line at the end of each file to avoid losses of titles in html
                f.write('\n')


if __name__ == "__main__":
    #clean_merged()
    merge_mds()

