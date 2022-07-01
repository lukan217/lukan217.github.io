---
title: Ridge和Lasso回归与代码实践
mathjax: true
date: 2021/5/23
tags:
- 机器学习
- 线性回归
categories: 
- 机器学习
---

线性回归作为一个非常经典的方法，被广泛应用于计量领域，用来解释变量对y的影响，但是在机器学习领域用纯粹的线性回归来做预测的好像就很少了，因为预测效果不怎么样，因此本文将对线性回归的两种改进方法做一个总结。

# Ridge & Lasso回归

在传统的线性回归中，是使用最小二乘法来估计参数的，通过最小化残差平方和来估计参数的，这个在机器学习领域也被称为损失函数：<br />$$LOSS_{ols}=\sum_{i=1}^{n}(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij})^2\\$$<br />这种最小二乘估计的方法被证明了具有最佳线性无偏估计（Best Linear Unbias Estimator, BLUE）的性质，所谓的最佳，就是方差最小，但这是在线性无偏估计的前提下，在有偏的情况下方差就不一定是最小了，设想一下，如果牺牲这个有偏的性质来使得方差变小呢，根据bias-variance trade-off，会不会有可能使得整体的预测误差进一步降低呢？<br />于是Ridge和Lasso的形式就被提出来了，通过牺牲传统ols回归中无偏的性质来使得方差降低，以寻求更低的预测误差，这两者的损失函数分别如下：<br />$$LOSS_{Ridge}=\sum_{i=1}^{n}(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij})^2+\lambda\sum_{j=1}^{p}\beta_j^2\\$$<br />$$LOSS_{Lasso}=\sum_{i=1}^{n}(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij})^2+\lambda\sum_{j=1}^{p}|\beta_j|\\$$<br />可以发现，这两个损失函数呢，就是在原来的ols的损失函数上加了一个系数惩罚项，因为我们求解时是让损失函数最小，加了后面这个惩罚项呢，会使得系数变小，这个$\lambda$就用来控制惩罚的力度，如果为0的话就和传统的线性回归没有差异了，如果是无穷大的话，那么所有的回归系数都会被弄到0，最后的所有的预测结果就是样本的均值了，但在实践中，我们可以通过交叉验证的方式调节$\lambda$的大小，选取最优的惩罚力度，就可以使得最终的预测误差达到最小。<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1621696524735-840f5da7-5fe4-4149-80a7-13ef9c3cbec6.png#align=left&display=inline&height=249&id=jCBob)<br />Ridge和Lasso这种加惩罚项的方式叫做正则化（Regularization），在机器学习的应用很广，比如神经网络中就有应用。因此，Ridge也被称为$L_2$正则化，后者被称为$L_1$正则化。<br />虽然两者的加的惩罚项看起来差不多，其实是有着非常大的区别的，具体表现为Lasso可以使得系数压缩到0，而Ridge则不会有这种效果，把系数压缩到0的话就可以起到降维和变量选择的作用，因此Lasso在高维的数据中表现更好。<br />那么为啥会有这样的差别呢，首先我们来看他们的惩罚项的形式，一个用的是平方的形式，另一个用的是绝对值的形式，我们把之前的那个损失函数转化成一个优化问题：<br />$$Ridge: \quad \min \sum_{i=1}^{n}(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij})^2 \quad s.t.\sum_{j=1}^{p}\beta_j^2 \le s\\
Lasso: \quad \min \sum_{i=1}^{n}(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij})^2 \quad s.t.\sum_{j=1}^{p}|\beta_j| \le s\\$$<br />假设只有两个系数，我们用几何的方式来表达这个优化问题，Ridge的约束条件是一个平方的形式，可行域就是一个圆，而Lasso的约束条件是绝对值的形式，可行域则是一个菱形，而目标函数在求解时，肯定是跟这个约束条件的可行域相切的，而Lasso由于他是一个菱形，那么他就更容易切到菱形的顶点，因此也会使得系数为0，而Ridge是一个圆，就不容易切到系数为0的地方，因此这就使得Lasso在压缩系数时会更倾向于压缩为0。<br />![image.png](https://cdn.nlark.com/yuque/0/2021/png/764062/1621697160270-8a07af21-ed57-4ad0-b2ca-de6eed23bcf8.png#align=left&display=inline&height=240&id=hG528)

# 代码实践

使用sklearn自带的波斯顿房价数据集做个试验，分别跑一遍Ridge和Lasso回归，并且通过交叉验证来选取$\lambda$，将之与线性回归进行对比，结果如下：

|         | MSE     |
| ------- | ------- |
| 线性回归    | 21.8977 |
| Ridge回归 | 21.7536 |
| Lasso回归 | 21.8752 |

可以发现，两者的预测效果较线性回归都有一定提升，其中Lasso回归提升较小，这是因为数据集的原因，只有13个变量，并且每个变量都make sense，因此效果就一般了，在高维的数据集中Lasso从理论上      讲应该就会有较好的表现了。<br />具体代码如下：

```python
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV,LassoCV
import numpy as np

boston = load_boston() # 导入波斯顿数据集
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f'线性回归MSE：{mean_squared_error(y_pred_lr, y_test)}')

ridge=RidgeCV(alphas=np.logspace(-5,5,11),cv=5) # lambda选择10的-5次方到5次方，五折交叉选择
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print(f'Ridge回归MSE：{mean_squared_error(y_pred_ridge, y_test)}')

lasso=LassoCV(alphas=np.logspace(-5,5,11),cv=5)# lambda选择10的-5次方到5次方，五折交叉选择
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print(f'Lasso回归MSE：{mean_squared_error(y_pred_lasso, y_test)}')
```

# 总结

本文对Ridge和Lasso回归做了一个总结，并通过一个简单数据集做了实践。在写的同时发现需要再去看和学习的东西很多，一个流程下来对于算法原理的理解更加透彻了，这对于搭建自己的知识体系是很有帮助的，希望以后能够坚持学完一个新的东西就写篇总结。
