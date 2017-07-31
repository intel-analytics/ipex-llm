import os
import argparse
import commands

def file_exists(file_path):
    if "No such file or directory" in commands.getoutput('ls %s' % file_path):
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create test text files from source')
    parser.add_argument(
        '-s', '--src', type=str,
        help="source directory")
    parser.add_argument(
        '-t', '--test', type=str,
        help="test directory",
    )

    args = parser.parse_args()
    src_dir = args.src
    test_dir = args.test
    if not file_exists(test_dir):
        commands.getoutput("mkdir %s" % (test_dir))

    count = 0
    for dir in os.listdir(src_dir):
        for file in os.listdir(src_dir + '/' + dir):
            if count < 10:
                # copy file
                dest_file_path = test_dir + '/'+ file
                if file_exists(dest_file_path):
                    continue
                else:
                    commands.getoutput('cp %s %s' % (src_dir + '/' + dir + '/' + file, dest_file_path))
                    count += 1
            else:
                count = 0
                break
