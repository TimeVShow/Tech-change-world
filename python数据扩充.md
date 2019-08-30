# python数据扩充

```python
# coding:utf-8
'''
作者：TimeVShow
time:2019/8/29
效果：找到标签文件中的所有的图片，对其进行水平翻转，垂直翻转等操作，
并保留改变后的图片，同时在标签集中写入图片的信息
'''

import os
from PIL import Image, ImageFilter
import random


def process_train_set(data_dir="./data_sets/cat_12", list_file_name="train_list.txt"):
    file_list_0 = os.path.join(data_dir, list_file_name_0)
    file_list = os.path.join(data_dir, list_file_name)
    with open(file_list, "r+") as flist:
        read_data = flist.read()
        flist.truncate()#清空文件
        for eachline in read_data.split('\n'):
            img_path,label = eachline.split('\t')
            label = label.split("\n")[0]
            img_full_path = os.path.join(data_dir, img_path)
            #如果文件中的图片不存在，直接跳过
            if(os.path.exists(img_full_path) == False):
                continue
            # 进行图像处理和保存，并写入文件
            # 水平翻转

            img_path_temp = horizontalFlip(img_full_path)
            line = img_path_temp.replace(data_dir + "/", "") + "\t" + label + "\n"
            flist.write(line)

            # 垂直翻转

            img_path_temp = verticalFlip(img_full_path)
            line = img_path_temp.replace(data_dir + "/", "") + "\t" + label + "\n"
            flist.write(line)

            # 色彩抖动
            for jitterring_type in range(1, 9):
                # 色彩抖动
                img_path_temp = jittering(img_full_path, jitterring_type=jitterring_type)
                print(img_path_temp)
                line = img_path_temp.replace(data_dir + "/", "") + "\t" + label + "\n"
                flist.write(line)

            # 旋转角度
            angles = [-30, -20, -10, 10, 20, 30]
            for index in range(len(angles)):
                # 旋转角度
                img_path_temp = rotate(img_full_path, rotate_angle=angles[index])
                print(img_path_temp)
                line = img_path_temp.replace(data_dir + "/", "") + "\t" + label + "\n"
                flist.write(line)

def modify_file_path(file_path, middle_name):
    # file_path = "D:/test/test.jpy"
    # middle_name = 'Hor'
    (filepath, tempfilename) = os.path.split(file_path)
    # print(filepath,tempfilename)
    (filename, extension) = os.path.splitext(tempfilename)
    # print(filename,extension)
    filename = filename + '_' + middle_name + extension
    file_path = os.path.join(filepath, filename)
    # file_path = 'D:/test/test_Hor.jpy'
    return file_path#返回新生成图片的绝对路径


# 水平翻转
def horizontalFlip(file_abs_path):
    middle_name = 'Hor'
    new_file_path = modify_file_path(file_abs_path, middle_name)
    img = Image.open(file_abs_path).convert('RGB')
    img.transpose(Image.FLIP_LEFT_RIGHT).save(new_file_path)#进行转化，并且保存图片（以下同）
    return new_file_path


# 垂直翻转
def verticalFlip(file_abs_path):
    middle_name = 'Ver'
    new_file_path = modify_file_path(file_abs_path, middle_name)
    img = Image.open(file_abs_path).convert('RGB')
    img.transpose(Image.FLIP_TOP_BOTTOM).save(new_file_path)
    return new_file_path


# 旋转角度
def rotate(file_abs_path, rotate_angle):
    if rotate_angle >= -45 and rotate_angle <= 45:
        i_randint = rotate_angle
    else:
        i_randint = random.randint(-45, 45)
    middle_name = 'Rotate%d' % (i_randint)
    new_file_path = modify_file_path(file_abs_path, middle_name)
    img = Image.open(file_abs_path).convert('RGB')
    img.rotate(i_randint).save(new_file_path)
    return new_file_path


# 随机抠图
def randCrop(file_abs_path):
    i_randint = random.randint(1, 10)
    middle_name = 'RandCrop_%d' % (i_randint)
    new_file_path = modify_file_path(file_abs_path, middle_name)

    img = Image.open(file_abs_path).convert('RGB')
    width, height = img.size
    # print(width, height)
    ratio = 0.88  # 0.8--0.9之间的一个数字
    left = int(width * (1 - ratio) * random.random())  # 左上角点的横坐标
    top = int(height * (1 - ratio) * random.random())  # 左上角点的纵坐标

    crop_img = (left, top, left + width * ratio, top + height * ratio)
    im_RCrops = img.crop(crop_img)
    im_RCrops.save(new_file_path)
    return new_file_path


# 色彩抖动
def jittering(file_abs_path, jitterring_type=0):
    img = Image.open(file_abs_path).convert('RGB')
    jitterring_type = int(jitterring_type)
    if jitterring_type >= 1 and jitterring_type <= 9:
        i_randint = jitterring_type
    else:
        i_randint = random.randint(1, 9)
    if i_randint == 1:
        # 高斯模糊
        middle_name = 'Jittering_%s' % ('GaussianBlur')
        new_file_path = modify_file_path(file_abs_path, middle_name)
        img.filter(ImageFilter.GaussianBlur).save(new_file_path)
    elif i_randint == 2:
        # 普通模糊
        middle_name = 'Jittering_%s' % ('BLUR')
        new_file_path = modify_file_path(file_abs_path, middle_name)
        img.filter(ImageFilter.BLUR).save(new_file_path)
    elif i_randint == 3:
        # 边缘增强
        middle_name = 'Jittering_%s' % ('EDGE_ENHANCE')
        new_file_path = modify_file_path(file_abs_path, middle_name)
        img.filter(ImageFilter.EDGE_ENHANCE).save(new_file_path)
    elif i_randint == 4:
        # 找到边缘
        middle_name = 'Jittering_%s' % ('FIND_EDGES')
        new_file_path = modify_file_path(file_abs_path, middle_name)
        img.filter(ImageFilter.FIND_EDGES).save(new_file_path)
    elif i_randint == 5:
        # 浮雕
        middle_name = 'Jittering_%s' % ('EMBOSS')
        new_file_path = modify_file_path(file_abs_path, middle_name)
        img.filter(ImageFilter.EMBOSS).save(new_file_path)
    elif i_randint == 6:
        # 轮廓
        middle_name = 'Jittering_%s' % ('CONTOUR')
        new_file_path = modify_file_path(file_abs_path, middle_name)
        img.filter(ImageFilter.CONTOUR).save(new_file_path)
    elif i_randint == 7:
        # 锐化
        middle_name = 'Jittering_%s' % ('SHARPEN')
        new_file_path = modify_file_path(file_abs_path, middle_name)
        img.filter(ImageFilter.SHARPEN).save(new_file_path)
    elif i_randint == 8:
        # 平滑
        middle_name = 'Jittering_%s' % ('SMOOTH')
        new_file_path = modify_file_path(file_abs_path, middle_name)
        img.filter(ImageFilter.SMOOTH).save(new_file_path)
    else:
        # 细节
        middle_name = 'Jittering_%s' % ('DETAIL')
        new_file_path = modify_file_path(file_abs_path, middle_name)
        img.filter(ImageFilter.DETAIL).save(new_file_path)

    return new_file_path
```