---
title: Git常用操作总结
tags:
- git
categories:
- 教程
---



最近工作上会用到Git，这里记录一下一些常用操作供参考！

# Git配置

1. 安装Git：[https://git-scm.com/](https://git-scm.com/)
1. 本地命令行生成密钥绑定GitHub账号
```shell
# 输入命令生成密钥对，替换成自己邮箱，然后一路回车
ssh-keygen -t rsa -C "youremail@example.com"

# 将生成的公钥打印出来复制，将这串文本复制粘贴到GitHub的Setting->SSH and GPG keys中
cat ~/.ssh/id_rsa.pub

# 输入命令检查是否绑定成功,输入yes后，如果出现Hi,xxx!则绑定成功
ssh -T git@github.com

```

3. 配置用户名和邮箱信息：
```shell
# 查看配置信息，一开始为空
git config --list

# 全局配置，对所有代码库生效
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
 

# 局部配置，只对当前的代码库有效
git config --local user.name "你的名字"
git config --local user.email "你的邮箱"
 
# 配置后，远程仓库提交的commit里对应的用户即为 user.name

```

# Git基本概念

1. 本地仓库：本地仓库上存放所有相关的文件，具体可分为工作区、暂存区和仓库区，工作区即项目文件夹下不包含`.git`文件夹的所有文件，暂存区和仓库区则在`.git`文件夹下
   1. 工作区：即我们工作的文件夹，在里面进行文件的增删改操作
   1. 暂存区：临时保存工作区上的改动，通过`git add`操作将工作区的修改同步到暂存区
   1. 仓库区：当执行`git commit`操作时，将暂存区上的所有变动同步到本地仓库
2. 远程仓库：GitHub/GitLab上保存的仓库，通过`git push`将本地仓库同步到远程仓库，也可以通过`git fetch/pull`将远程仓库同步到本地仓库

![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1630159880578-b47a2dae-9236-4c17-8de3-f4d7155ac69f.png)

# Git基本操作

## 创建版本库
创建版本库有两种方式，一种是将本地的文件夹直接变成一个git仓库，另一种是直接将远程的仓库克隆到本地
```shell
git init # 将本地文件夹变为一个git仓库
git clone <url> #将远程仓库克隆到本地
```

## 修改与提交操作
```shell
git add <file> # 将单个文件从工作区添加到暂存区
git add . # 将所有文件添加到暂存区
git commit -m "messenge" # 将暂存区文件提交到本地仓库
git status # 查看工作区状态，显示有变更的文件。
git diff # 比较文件的不同，即暂存区和工作区的差异。
```

## 远程操作
```shell
git push origin master # 将本地的master分支推送到远程对应的分支
git pull  # 下载远程代码并合并，相当于git fetch + git pull
git fetch	# 从远程获取代码库，但不进行合并操作

git remote add origin <url> # 将远程仓库与本地仓库关联起来
git remote -v # 查看远程库信息

```

## 撤销与回退操作
撤销操作：当修改了工作区/暂存区的文件，但是还没有commit时，想要撤销之前的操作：
```shell
# 场景1：当你改乱了工作区某个文件的内容，但还没有add到暂存区
git checkout <file> # 撤销工作区的某个文件到和暂存区一样的状态

# 场景2：当乱改了工作区某个文件的内容，并且git add到了暂存区
git reset HEAD <file> # 第1步，将暂存区的文件修改撤销掉
git checkout <file> # 第2步，将工作区的文件修改撤销掉

# 场景3：乱改了很多文件，想回到最新一次提交时的状态
git reset --hard HEAD # 撤销工作区中所有未提交文件的修改内容

```
回退操作：当已经进行了commit操作，需要回退到之前的版本：
```shell
git reset --hard HEAD^ # 回退到上次提交的状态
git reset --hard HEAD~n # 回退到n个版本前的状态
git reset --hard HEAD commitid # 回退到某一个commitid的状态
```

# Git分支管理
git的最强大之处就在于分支管理了，具体有两种应用场景：

1. 多人协作：每个人都基于主分支创建一个自己的分支，在分支上进行开发，然后再不断将写好的代码合并到主分支
1. 自己修复bug/增加feature：创建一个bug分支或者feature分支，写好代码后合并到自己的分支然后删除bug/feature分支
```shell
git branch <name> # 创建分支
git checkout <name> # 切换到某个分支
git checkout -b newtest # 创建并切换到新分支，相当于同时执行了以上两个命令
git merge <name> # 合并某个分支到当前分支中，默认fast forward
git branch -a # 查看所有分支
git branch -d <name> # 删除分支

```

# Git多人协作
多人协作在同一个分支上进行开发的工作模式：

1. 首先，可以试图用`git push origin <branch-name>`推送自己的修改；
1. 如果推送失败，则因为远程分支比你的本地更新，需要先用`git pull`试图合并；
1. 如果合并有冲突，则解决冲突，并在本地提交；
1. 没有冲突或者解决掉冲突后，再用`git push origin <branch-name>`推送就能成功！
2. 如果`git pull`提示`no tracking information`，则说明本地分支和远程分支的链接关系没有创建，用命令`git branch --set-upstream-to <branch-name> origin/<branch-name>`。<br />

# 参考

1. [Git教程 | 菜鸟教程](https://www.runoob.com/git/git-tutorial.html)
1. [Git教程 - 廖雪峰的官方网站](https://www.liaoxuefeng.com/wiki/896043488029600)
