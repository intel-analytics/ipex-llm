import sys
import random
jobs=['Developer', 'Engineer', 'Researcher']
output_file = sys.argv[1]
num_lines = int(sys.argv[2])
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
