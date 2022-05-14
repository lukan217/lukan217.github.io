---
title: 云服务器配置远程jupyter notebook环境
tags:
- jupyter notebook
- 服务器
categories:
- 计算机
---

去年年初疫情，阿里云搞了一个在家实践的活动，就免费领了半年的云服务器，从此打开了新世界的大门，比如写一些脚本在挂服务器上跑一些代码，搭一个网站，还有一个就是可以在服务器上搭一个jupyter notebook的环境，通过网址就可以直接打开notebook写代码了，适合方便快速地写一些小型的代码，或者在手头的电脑没有anaconda环境时直接使用，甚至用ipad或者手机也可以写，大致的效果如下：

1. 通过网址随时随地都能打开编程
1. 配置了适合编程的主题色调
1. 加入了插件补全功能

![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1624879339187-e21ea69c-5a13-4f9e-9a78-278e3a86edb6.png#height=365&id=dR0gt&)<br />前几天因为折腾自己的服务器环境给搞崩了，数据库出了点问题，所以只能重装系统，导致jupyter notebook又要重装一遍，然后几个月后服务器到期，估计又要重新配一遍环境，就索性写一篇教程，供自己日后和有需要的人参考。


# 云服务器选购
首先需要选购一个云服务器，推荐腾讯云或者阿里云，有学生认证的话一年大概100左右，操作系统推荐是用目前主流的两个Linux发行版，ubuntu和cent OS，两个系统在一些安装软件的命令上会有小差异，我这里用的是ubuntu。


# 安装Anaconda
在买好云服务器后，就通过ssh连接，就可以用命令行进行操作了，首先第一步是安装anaconda，先要下载anaconda的安装包，输入命令：
```shell
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
```
下载好后直接安装：
```shell
bash Anaconda3-2021.05-Linux-x86_64.sh
```
![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1624898431857-7b2a0ec6-2970-4c6f-9af3-07dc336f9a3d.png#height=72&id=DlHXa&)<br />会弹出这样一个界面，直接一直回车，然后输入yes继续回车，等待安装完成即可，安装完成会有这样一个界面，就代表安装完成了<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1624898994446-2e991619-5f16-4b5f-bd61-3292b2fc53f7.png#height=126&id=P7Dy3)


# 配置jupyter notebook环境
接下来就可以配置jupyter notebook环境了，首先需要生成一个配置文件，输入命令：
```shell
jupyter notebook --generate-config
```
因为服务器的安全性，配置远程访问是需要设置一个密码的，输入命令，生成密钥：
```shell
jupyter notebook password
```
输入两次密码，这里就会生成一个密钥放在用户文件夹的.jupyter文件夹下，和刚刚的配置文件路径一样，这两个文件会自动关联起来，在修改配置文件的时候就不需要加跟密钥相关的命令了。<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1624899427258-a0a09979-4f70-4935-9933-d058534df757.png#height=52&id=i9wwI)<br />接下来就可以直接修改刚刚生成的那个配置文件了，使用vim打开，输入命令：
```shell
vim ~/.jupyter/jupyter_notebook_config.py
```
按键盘的i键进入编辑模式，直接在开头添加以下内容：
```shell
c.NotebookApp.ip='*' # 代表任意ip是都可以访问jupyter
c.NotebookApp.notebook_dir='/home/ubuntu/jupyter' # notebook的工作目录，可以自己的实际情况修改，注意要确保目录存在
c.NotebookApp.open_browser = False # 不打开浏览器
c.NotebookApp.port =8888  #可自行指定一个端口, 访问时使用该端口
```
按Esc键退出编辑模式，然后输入:wq保存即可。


# 开启远程访问
我们在上一步中指定了端口为8888，也让所有ip都能够访问这个端口了，但是在云服务器中还需要把这个端口开启起来，以腾讯云为例，进入安全组中，添加入站规则，按如下设置，然后在出站规则里点击一键放通，入站规则和出站规则都需要配置好<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1624900683679-df0724ff-dcb9-4091-bf70-b9cb926120f2.png#height=205&id=p7ggP)<br />接下来就可以将jupyter notebook打开了，不过我们需要能够将notebook一直在后台挂着，所以这里就输入这个命令：
```shell
nohup jupyter notebook > jupyter.log 2>&1 &
```
这里nohup（no hang up）是不挂起的意思，用于在系统后台不挂断地运行命令，退出终端不会影响程序的运行，最后面的**&**是让命令在后台执行，终端退出后命令仍旧执行，> jupyter.log 2>&1是输出日志的意思，把命令的输出和错误都写到jupyter.log这个文件中，方便监控。<br />接下来我们在浏览器中输入：服务器公网ip:端口号，即可访问jupyter，如图所示，再输入刚刚设置的密码就行了<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1624903214506-5212ba31-e39b-497e-bbb4-200892853bb9.png#height=390&id=y6mOt)<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1624903977925-09aa1daf-f1f3-40e2-ad21-8d1f42269d66.png#height=229&id=SPHn7)


# 装代码补全插件与更换主题
在上一步中，我们已经配置好了一个可以远程访问的jupyter notebook，但是呢，这个notebook的主题是默认的，白色太亮眼不适合编程，而且，默认的jupyter notebook也没有补全代码的功能，所以就通过插件的方式来解决这两个问题。



## 补全代码插件
依次执行以下命令：
```shell
pip install jupyter_contrib_nbextensions 
jupyter contrib nbextension install --user
pip install jupyter_nbextensions_configurator
jupyter nbextensions_configurator enable --user
```
这样插件就装好了



## 更换主题
首先安装jupyterthemes：
```python
pip install jupyterthemes
```
jupyterthemes是一个为jupyter notebook设置主题的插件，可以在github上查看他们的使用手册，<br />
这里推荐自己的一套配置方案，在命令行输入：

```shell
jt -t chesterish -f roboto -fs 12 -ofs 105 -dfs 95 -lineh 150
```




## 重启jupyter notebook
如果你的notebook在运行中，就需要重启一下才能使得上一步的修改生效，首先找到运行jupyter notebook的进程id，然后杀掉这个进程，再重启就可以了
```shell
ps -aux|grep jupyter
sudo kill -9 进程id
nohup jupyter notebook > jupyter.log 2>&1 &
```
可以发现，现在界面已经跟刚才的不一样了：<br />

![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1624904807402-493d550d-ae8c-4963-82bf-69a99fff310e.png#height=435&id=gz8pT)<br />

然后还需要在Nbextensions中开启下Hinterland，也就是我们的补全插件<br />
<br />![](https://cdn.nlark.com/yuque/0/2021/png/764062/1624384565647-dab6f490-7909-4501-a903-1cab72df72e0.png?x-oss-process=image%2Fresize%2Cw_1404#height=722&id=AIyRA&originHeight=722&originWidth=1404&originalType=binary&ratio=1&status=done&style=none&width=1404)

然后就大功告成了，有一个养眼的界面和补全代码的功能，就可以随时随地都能用搭建好的这个环境写一些代码了

![](https://cdn.nlark.com/yuque/0/2021/png/764062/1624904949851-5191b2d7-4550-423d-b53c-71f609b93a10.png#height=404&id=QiYzW)
