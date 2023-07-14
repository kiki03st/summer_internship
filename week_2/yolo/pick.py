import os
import shutil

img = os.getcwd() + '/img/'
label = os.getcwd() + '/img_data/'
move = os.getcwd() + '/move/'
#f = open('valid.txt', mode = 'w')
valid_list = []
for i in os.listdir(img):
    for j in os.listdir(label):
        if j == 'classes.txt': continue
        elif i[:-4] == j[:-4]:
#            valid_list.append('/mnt/c/Users/ensung/Desktop/darknet/data/valids/' + i)
            shutil.copyfile(img + i, move + '/images/' + i)
            shutil.copyfile(label + j, move + '/labels/' + j)
#f.write('\n'.join(valid_list))
#f.close()
print(os.listdir(move + '/images/'))
print(os.listdir(move + '/labels/'))
