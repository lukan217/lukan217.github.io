---
title: 前向分步回归Forward Stagewise Regression原理及Python实现
mathjax: true
date: 2021/5/4
tags:
- 机器学习
- 回归算法
- boosting
categories:
- 机器学习算法
---
# 前言
最近偶然接触到一种回归算法，叫做前向分布回归（Forward Stagewise Regression），注意这不是那个向前逐步回归（Forward stepwise regression），stepwise和stagewise，还是有区别的，网上关于他的介绍非常少，中文社区基本就没怎么看到了，就顺手写一下吧，算法的思想来源于boosting，理解这个也有助于之后对各种树模型的boosting算法的学习。

# 算法原理
这个算法的思想与boosting类似，每次迭代时都挑选出一个最优的变量来拟合残差，具体步骤如下：

1. 首先将截距项$\beta _0$设置为$\bar{y}$，所有的自变量系数$\beta$都设为0，残差项设置为$r=y-\bar y$
1. 挑选出与残差项最相关的自变量$x_j$
1. 更新$\beta _j$的值：，其中$\delta_j=\epsilon \times \text{sign}[\langle x_j,r \rangle]$，这个$\text{sign}[\langle x_j,r \rangle]$代表相关性的正负，$\epsilon$代表步长。再更新下残差项的值：$r=r-\delta_j x_j$
1. 重复步骤2，3，直到达到最大迭代次数或者所有的变量都与残差项无关。
![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1620114719031-b1f1c1e8-155e-4258-a114-0d63a13b6a42.png#clientId=ucf785b6c-7490-4&from=paste&height=259&id=uaa9a13b9)
这个算法的优点在于与Lasso回归有着异曲同工之妙，通过选择合适的迭代次数和步长，可以使得部分变量的系数压缩为0，就可以起到变量选择和降低方差的作用，因此在高维数据的场景下会有较好的表现，再偷一张《The Elements of
Statistical Learning》的变量系数路径图来说明这一点，左图的横轴为Lasso的L1范式，右图的横轴为前向分布回归的迭代次数，可以看到，变量系数的压缩路径大体上是一致的。
![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1620118711730-c7178912-4b0e-447d-8355-2bdae92fcc77.png#clientId=ucf785b6c-7490-4&from=paste&height=330&id=uff065e10)

# Python实现
用波斯顿房价的数据集来做个测试，将迭代次数设为2000的时候，mse要略小于线性回归：
![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1620121931201-e7594c64-9878-47d3-a851-0285bf12f751.png#clientId=ucf785b6c-7490-4&from=paste&height=44&id=j2U1Z)
因为这个数据集只有13个变量，而且每个变量都很重要，所以前向分布回归的优势并没有很明显，不过通过调参效果还是可以比普通的线性回归好那么一点，代码如下：

```python
import numpy as np

class ForwardStagewise():
    def __init__(self, eps=0.01, max_iter=1000):
        # 初始化两个参数，eps步长和max_iter迭代次数
        self.eps = eps
        self.max_iter = max_iter

    def fit(self, X, y):
        # 训练模型
        X = np.asarray(X) # 将X，y转化为数组形式
        y = np.asarray(y)
        X_mean = np.mean(X, axis=0) # 标准化
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        self.y_mean = np.mean(y) # 截距项，也就是y的平均
        residual = y - self.y_mean # 初始化残差项
        x_num = np.shape(X)[1] # 变量数
        self.beta = np.zeros((x_num)) # 用来存储每一次系数更新的数组
        self.betas = np.zeros((self.max_iter, x_num))  # 用来存储每一迭代的系数
        for i in range(self.max_iter):
            c_hat = 0
            sign = 0
            best_feat = -1
            for j in range(x_num):
                c_temp = X[:, j].T.dot(residual) # 用来表示x与残差项的相关性
                if abs(c_temp) > c_hat:
                    c_hat = abs(c_temp)
                    sign = np.sign(c_temp)
                    best_feat = j
            self.beta[best_feat] += sign * self.eps # 更新系数
            residual -= (self.eps * sign) * X[:, best_feat] # 更新残差项
            self.betas[i] = self.beta
        return self

    def predict(self, X):
        # 预测
        X = np.asarray(X) # 先标准化
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_test = (X - X_mean) / X_std
        y_pred = X_test.dot(self.beta) + self.y_mean
        return y_pred


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    boston = load_boston() # 导入波斯顿数据集
    X = boston.data
    y = boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    fs = ForwardStagewise(eps=0.01, max_iter=2000)
    fs.fit(X_train, y_train)
    y_pred = fs.predict(X_test)
    print(f'前向逐步回归MSE：{mean_squared_error(y_pred, y_test)}')
    lg = LinearRegression()
    lg.fit(X_train, y_train)
    y_pred_lg = lg.predict(X_test)
    print(f'线性回归回归MSE：{mean_squared_error(y_pred_lg, y_test)}')
```

# 总结
前向分布回归和Lasso回归本质上其实差不多，而且两者好像都是最小角回归（Least angle regression）的一个变种，具体可以参见ESL这本书（太难了我看不懂），这两张回归算法都能起到压缩系数和变量选择的作用，但是前向分布回归的计算效率比较差，所以Lasso似乎更为人熟知，不过前者为我们学习boosting相关算法提供了一个不错的切入点。
