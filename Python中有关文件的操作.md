# Python中有关文件的操作

## 1.文件路径的书写

```python
import os
#这里返回的值是字符串
path = os.path.join(root_dir,file_dir)
```

## 2.列出文件夹中的文件

```python
import os
#这里返回的值是一个列表
list = os.listdir(rootdir)
```

## 3.打开文件并进行读写

```python
#根据取mode的不同会有对应的不同形式，具体看下表格
file = open(file_path,mode)
#直接读取的是不可操作的
#file.read()返回字符串类型
#for line in file逐行读取file文件
```

|             模式             |  r   |  r+  |  w   |  w+  |  a   |  a+  |
| :--------------------------: | :--: | :--: | :--: | :--: | :--: | :--: |
|              读              |  +   |  +   |      |  +   |      |      |
|              写              |      |  +   |  +   |  +   |  +   |  +   |
|             创建             |      |      |  +   |  +   |      |      |
|             覆盖             |      |      |  +   |  +   |      |      |
|  指针在开始（从头开始写入）  |      |      |      |      |      |      |
| 指针在结尾（从尾部开始写入） |      |      |      |      |  +   |  +   |