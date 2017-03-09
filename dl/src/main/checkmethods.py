import os
python_path = "python/nn"
scala_path = "scala/com/intel/analytics/bigdl/nn"
filesList = os.listdir(scala_path)
allMethods = []
for eachFile in filesList:
    methodName = eachFile.split('.')[0]
    allMethods.append(methodName)
print "there are",len(allMethods),"in total"
layer = open(python_path+"/layer.py")
existMethods = []
for eachLine in layer:
    elems = eachLine.strip().split()
    #print elems
    if len(elems)>0 and elems[0] == "class":
        leftmost = elems[1].find('(')
        existMethodName = elems[1][:leftmost]
        existMethods.append(existMethodName)

for v in allMethods:
    if v not in existMethods:
        print v

print [v for v in allMethods if v in existMethods]
print existMethods
