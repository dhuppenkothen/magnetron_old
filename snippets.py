__author__ = 'daniela'

import rxteburst
import shutil

allflag = 0
for o in oldfiles:
    osplit = o.split("/")
    flag = 0
    for n in newfiles:
        nsplit = n.split("/")
        print("nsplit[-1]: " + str(nsplit[-1]))
        print("osplit[-1]: " + str(osplit[-1]))
        if nsplit[-1] == osplit[-1]:
            flag +=1
        else:
            flag += 0
            print("flag: " + str(flag))
            #shutil.copy(o, "../properprior/")
            #print("Missing file " + str(o) + ", copying to directory properprior")
    if flag == 0:
        print("Missing file " + str(o) + ", copying to directory properprior")
        shutil.copy(o, "../properprior/")
