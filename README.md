# MNIST_Try
利用简单的卷积层构建，结合MNIST数据集，实现数字识别任务

并且提供网页版分析，更深层次体会卷积神经网络的神奇！

## 开始
conda环境配置

`conda create -n MNIST_try python=3.12`

然后看看需要什么库，代码里有（作者很懒）

可以`pip install xxx`

或者`conda install xxx`一下

## 运行

当项目拉取到本地后

`git clone https://github.com/Alittlepiggy/MNIST_Try.git`

你可以看到这样的文件格式，然后即可玩一玩了！

```
Root
  -- data(将代码中download改成True会出现)
  -- templates
    -- index.html
  -- app.py(网页尝试)
  -- weights
    --......
  -- train_and_test.ipynb(训练和data下载)
```
## 介绍

app.py文件运行可以用命令

`python app.py`

然后访问浏览器地址

`http://127.0.0.1:5000`

data数据集下载一次即可，之后就可以将下载关闭！

app.py中涉及到weight的选取，这与训练时表现有关，请自行选取！


