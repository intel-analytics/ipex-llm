import sys
import os
import random

#define safe directory
safe_dir = '/home/'

def is_path_within_safe_directory(requested_path):
    # Append '/' to the safe directory to account for the check when requested_path is the same as safe_dir.
    safe_dir_with_slash = safe_dir if safe_dir.endswith('/') else safe_dir + '/'
    return os.path.commonprefix((os.path.realpath(requested_path), safe_dir_with_slash)) == safe_dir_with_slash

jobs=['Developer', 'Engineer', 'Researcher']
output_file = sys.argv[1]
num_lines = int(sys.argv[2])

if not is_path_within_safe_directory(output_file):
    print("Bad user! The requested path is not allowed.")
else:
    with open(output_file, 'wb') as File:
        File.write("name,age,job\n".encode())
        cur_num_line = 0
        num_of_developer_age_between_20_and_40 = 0
        while(cur_num_line < num_lines):
            name_length = random.randint(3, 7)
            name = ''
            for i in range(name_length):
                name += chr(random.randint(97, 122))
            age=random.randint(18, 60)
            job=jobs[random.randint(0, 2)]
            if age <= 40 and age >= 20 and job == 'Developer':
                num_of_developer_age_between_20_and_40 += 1
            line = name + ',' + str(age) + ',' + job + "\n"
            File.write(line.encode())
            cur_num_line += 1
        print("Num of Developer age between 20,40 is " + str(num_of_developer_age_between_20_and_40))
    File.close()
