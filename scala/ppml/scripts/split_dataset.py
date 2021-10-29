
from csv import reader,writer
import numpy as np
import sys


def split_dataset(fname,ind_arg):
  with open(fname, 'r') as read_obj:
    header=np.array(["ID","Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"])
    ind_1=[0]+[ind+1 for ind in ind_arg]
    ind_2=ind_2=[0]+[ind for ind in range(1,len(header)) if ind not in ind_1 ]

    csv_reader = reader(read_obj)
    with open("./diabetes-1.csv", "w") as write_obj_1:
      with open("./diabetes-2.csv", "w") as write_obj_2:

        csv_writer_1 = writer(write_obj_1, delimiter=',')
        csv_writer_2 = writer(write_obj_2, delimiter=',')

       

        # pass the file object to reader() to get the reader object
        
        # Iterate over each row in the csv using reader object
        for i,row in enumerate(csv_reader):
            # row variable is a list that represents a row in csv
            if i==0:
              if not row[0].isdigit():
                header=np.array(["ID"]+row)
                csv_writer_1.writerow(header[ind_1])
                csv_writer_2.writerow(header[ind_2])
              else:
                csv_writer_1.writerow(header[ind_1])
                csv_writer_2.writerow(header[ind_2])
                row=np.array([int(i+1)]+[float(d) for d in row])
                csv_writer_1.writerow(row[ind_1])
                csv_writer_2.writerow(row[ind_2])
            else:
              row=np.array([int(i+1)]+[float(d) for d in row])

              csv_writer_1.writerow(row[ind_1])
              csv_writer_2.writerow(row[ind_2])




def main():
  ind_arg=[0,1,2,3,8]
  fname=input("please type data file name:")
  if len(fname)==0:
      fname="pima-indians-diabetes.csv"
  if len(sys.argv)>1:
    ind_arg=[int(col) for col in sys.argv[1:]]
  split_dataset(fname,ind_arg)

if __name__ == "__main__":
    main()
