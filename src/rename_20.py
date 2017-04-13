import os
import numpy as np
import shutil

folder_path = '/home/ubuntu/instanceMatch/data20'
imgName = os.listdir(folder_path)
#print(len(b));
tim = [9,10,11,12,13,14,15,16,17,18]
new_folder = '/home/ubuntu/instanceMatch/data_20_renamed'
count = 0
for filename in imgName:
    a = filename.split(os.extsep)
    b = a[0].split(' ')
    c = b[1].split(':')
    count = count + 1
    #print(type(c[0]))
    for one_time in tim:
        if(int(c[0]) == one_time):
            if((one_time % 2) == 1):
                time_folder = str(one_time);
                #print(time_folder)
                source = os.path.join(folder_path,filename)
                #print(source)
                destination = new_folder + '/' + time_folder +  '/'  + str(count) + '.jpg' 
                #print(destination)
                shutil.copyfile(source,destination)
            else:
                time_folder = str(one_time - 1)
                source = os.path.join(folder_path,filename)
                destination = new_folder + '/' + time_folder + '/' + str(count) + '.jpg' 
                shutil.copyfile(source,destination)







    


