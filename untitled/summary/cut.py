import os
import numpy as np
import cv2

noncov_path="C://deep_learning//COVID DATA//COVID DATA//CT_NonCOVID"
cov_path="C://deep_learning//COVID DATA//COVID DATA//CT_COVID//CT_COVID"
nfile_name=[]
cfile_name=[]
nimage_path=[]
cimage_path=[]
for filename in os.listdir(noncov_path):
    nfile_name.append(filename)
for filename in nfile_name:
    nimage_path.append(os.path.join(noncov_path, filename))

for filename in os.listdir(cov_path):
    cfile_name.append(filename)
for filename in cfile_name:
    cimage_path.append(os.path.join(cov_path, filename))


savepath='E://deep_learning//COVID DATA//cut//0//'
savep='E://deep_learning//COVID DATA//cut//1//'
ss='E://deep_learning//COVID DATA//cut//2//'
sss='E://deep_learning//COVID DATA//cut//3//'



count = 0
for ipath in nimage_path:
    img = cv2.imread(ipath, 0)
    res = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    [a1, a2] = np.split(res, 2, axis=1)
    s1=savepath+nfile_name[count]
    s2=savep+nfile_name[count]
    cv2.imwrite(s1, a1)
    cv2.imwrite(s2,a2)
    count = count + 1
count = 0
for ipath in cimage_path:
    img = cv2.imread(ipath, 0)
    res = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    [a1, a2] = np.split(res, 2, axis=1)
    s1=ss+cfile_name[count]
    s2=sss+cfile_name[count]
    cv2.imwrite(s1, a1)
    cv2.imwrite(s2,a2)
    count = count + 1