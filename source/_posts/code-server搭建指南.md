---
title: code-server搭建指南
tags:
- 服务器
categories:
- 计算机
---

虽然自己之前搞了一台服务器，也在服务器上[部署了jupyter notebook](https://zhuanlan.zhihu.com/p/384888122)，但是仍有两个痛点没有解决：

1. 服务器部署了一些代码，有时候需要修改，通过vim直接修改是不现实的，因为没有补全高亮，改起来很麻烦，只能本地改好再上传上去
2. 虽然部署了jupyter，能够实现一些简单的代码编辑需求，但是仅限于ipynb，其他文件无法编辑查看，并且补全功能十分鸡肋

因此，为了能够在浏览器里面得到和本地编程一样丝滑的体验，最终决定部署一个code-server，也就是web版的vscode，实测体验和本地的vscode没有任何区别，效果如下：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1657447452075-701b98ea-8435-4da3-b86a-5181aac0797d.png#clientId=u7f73bf32-1eb1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=841&id=u78da3e34&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1262&originWidth=2560&originalType=binary&ratio=1&rotation=0&showTitle=false&size=83496&status=done&style=none&taskId=ucf3b3635-f4a1-4f4e-ab2c-a5e8c1e0bd3&title=&width=1706.6666666666667)

<a name="FavUM"></a>

# 搭建过程

<a name="wJ0di"></a>

## 准备

1. 一台云服务器
2. 一个经过公安部备案的域名

为什么要域名呢？因为我经常需要用jupyter notebook，但是这玩意在code-server里面由于安全性的原因需要通过https才能打开，但是通过ip地址是没办法走https的，因此就需要一个域名，而且是要经过备案的，不然没法访问。当然，如果你不需要用到jupyter notebook可以直接跳过这个步骤。<br />具体申请流程如下，以腾讯云为例：

1. 购买一个域名：[https://console.cloud.tencent.com/domain](https://console.cloud.tencent.com/domain)
2. 为域名备案，走完整套流程大概要2周：[https://console.cloud.tencent.com/beian](https://console.cloud.tencent.com/beian)
3. 最后一步，添加DNS解析：[https://console.dnspod.cn/](https://console.dnspod.cn/)

主机记录可以填一个前缀，比如code，最后就是通过code.xxx.com来访问，记录值填写服务器公网ip，这样就在浏览器里面输入域名就会自动解析到服务器的地址了<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1657434255367-9c6b978a-c2c9-4976-86d6-8588c85b01d1.png#clientId=u7f73bf32-1eb1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=355&id=u5a4e0ffc&margin=%5Bobject%20Object%5D&name=image.png&originHeight=533&originWidth=2034&originalType=binary&ratio=1&rotation=0&showTitle=false&size=93394&status=done&style=none&taskId=uc9c8c788-7296-44dd-bfb1-93ba8620160&title=&width=1356)
<a name="BqcT9"></a>

## code-server配置

完成准备步骤后，就可以配置code-server了，安装步骤也很简单，依次输入以下命令就行了：

```shell
curl -fsSL https://code-server.dev/install.sh | sh 
```

如果上面的命令因为墙的原因下载不了，就只能通过本地下载安装包，传到服务器，再手动安装，这里以ubuntu为例：

```shell
sudo dpkg -i code-server_4.5.0_amd64.deb
```

然后输入命令行输入`code-server`, 会生成一个本地配置文件，ctrl+C关闭，再去改配置文件：

```shell
vim ~/.config/code-server/config.yaml
===============
bind-addr: 0.0.0.0:8080 # 如果没域名需要改成这个
auth: password
password: 123456
cert: false
===============
code-server
```

这时候浏览器输入：公网ip:8080应该就能访问了
<a name="HBU8I"></a>

## 配置https访问

完成以上的操作，code-server的基本配置就完成了，但是之前说过，这样是不完整的，因为没有域名，并且没有https，很多操作进行不了，所以建立弄一个备案好的域名，然后根据官网给的操作说明，配置nginx和用Let's Encrypt生成证书，依次进行以下操作：

```shell
# 安装nginx并配置
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
vim /etc/nginx/sites-available/code-server
# 填入以下内容，域名记得改一下
===========================================
server {
    listen 80;
    listen [::]:80;
    server_name mydomain.com;

    location / {
      proxy_pass http://localhost:8080/;
      proxy_set_header Host $host;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection upgrade;
      proxy_set_header Accept-Encoding gzip;
    }
}
============================================
sudo ln -s ../sites-available/code-server /etc/nginx/sites-enabled/code-server
# 为域名生成证书，最后那个是你的邮箱
sudo certbot --non-interactive --redirect --agree-tos --nginx -d mydomain.com -m me@example.com
```

<a name="eaemE"></a>

## 配置守护进程

```shell
vim /usr/lib/systemd/system/code-server.service
# 输入以下配置
=========================
[Unit]
Description=code-server
After=network.target

[Service]
Type=exec
Environment=HOME=/root
ExecStart=/usr/bin/code-server
Restart=always
=========================
# 然后就可以通过以下命令来启动和关闭code-server服务了
# start code-server
systemctl start code-server
# stop code-server
systemctl stop code-server
# code-server status
systemctl status code-server
```

这样就基本配置成功了，之后再根据自己的需要装上插件，换下主题，就完全和本地的vscode没啥区别，可以随时随地在浏览器连接服务器进行编程了！
