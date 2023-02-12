---
title: Kaggle Otto推荐系统比赛TOP方案总结
mathjax: true
tags:
- kaggle
- 推荐系统
categories:
- 比赛
---
最近Otto的比赛完结了，总结学习一下奖金区的方案，顺便在文末放自己一份单模在榜单上能排大概22名的代码

# 赛题介绍

- 赛题名称：[OTTO – Multi-Objective Recommender System](https://www.kaggle.com/competitions/otto-recommender-system)
- 赛题简介：本次比赛的目标是预测电子商务点击量、购物车添加量和订单。您将根据用户会话中的先前事件构建一个多目标推荐系统。
- 数据：训练数据包含4周的电子商务会话信息，每个session（可以理解为用户）中包含用户交互过的多个aid（商品），每个aid都有用户交互的行为（clicks,carts,orders）以及发生的时间戳，测试数据包含未来一周session按照均匀分布随机截断的数据，需要预测测试session中最后一个时间戳之后三种交互行为(clicks， carts，orders)可能对应的aid
- 评价指标：对每种交互类型的加权recall@20:
   - $\textit{score}=0.10\cdot R_{clicks}+0.30\cdot R_{carts}+0.60\cdot R_{orders}$
   - $R_{t ype}=\dfrac{\sum_{i}^{N}\left|\{\text{predicted aids}\}_{t,type}\cap\{\text{ground truth aids}\}_{i,type}\right|}{\sum_{i}^{N}\min\left(20,\left|\{\text{ground truth aids}\}_{i,tye}\right|\right)}$

整体比赛还是比较难的，一方面数据量大，需要较大的内存，同时需要懂的=各种优化操作，另一方面特征很少，用户和商品特征几乎没有，需要自行构建较多的相似度特征才能取得比较好的成绩。

# 1st Place Solution

- 链接：[https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022](https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022)
- 方案亮点：生成了1200个候选，并且使用一个精心设计的NN模型来进行召回，这个应该是在top方案里面唯一使用NN模型的

## 召回阶段
召回了1200个候选，召回策略包括：

1. session内交互过的aid
2. 共同访问矩阵
   1. 构建了多个版本，分别对类别以及时间进行不同的加权
   2. 像beam search一样多次从共同访问矩阵中进行召回
3. NN召回
   1. 构建了多个版本的NN模型来召回候选以及生成特征，NN的架构是MLP或者Transformer，具体方式如下：
   2. 在训练阶段，将session进行切分，一部分作为x_aids, 一部分作为y_aids，x_aids与一些时间特征还有拼接后输入一个NN，然后pooling后得到一个session embedding，同时，为了能够输出不同类型的embedding，将想要得到的类型也作为一个特征输入到NN中，然后y_aids则作为负样本，过embedding层后与session embedding计算余弦相似度，然后将计算的相似度取平均和最小值加和除以2，同时，也采样一些负样本，得到负样本的embedding后与session embedding计算余弦相似度后取top k个，最后计算 cross entropy损失。
   3. 在测试阶段，对于每一个session，将session内的所有aid丢进去计算一个session embedding，然后把所有的aid与session embedding计算余弦相似度后取top k作为最终的召回结果。

不得不说这个NN的训练方式设计的真的很巧妙，流程图如下：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1676140497214-d77571c3-0428-4271-a1fc-c3706d4dc714.png#averageHue=%23fbfafa&clientId=u80392229-df4f-4&from=paste&id=u99e9652b&name=image.png&originHeight=976&originWidth=1816&originalType=url&ratio=2&rotation=0&showTitle=false&size=150055&status=done&style=none&taskId=uc81087a6-3e89-4bc1-8dcc-050a272f709&title=)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1676140504832-5d5bf01c-f5aa-43cf-887a-b95836a9f3b8.png#averageHue=%23fcfcfc&clientId=u80392229-df4f-4&from=paste&id=ue687111c&name=image.png&originHeight=1002&originWidth=1804&originalType=url&ratio=2&rotation=0&showTitle=false&size=104807&status=done&style=none&taskId=u57e04ce1-e9fc-4157-932b-5f6c6d6811a&title=)

## 排序阶段

### 特征工程

1. session特征: 长度、aid重复率、最后一个aid与倒数第二个aid的时间差
2. aid特征：aid的热门程度（使用多个时间窗进行加权），不同行为类型的比率等
3. session与aid交互特征
   1. 共同访问矩阵的排名
   2. NN模型生成的余弦相似度
   3. session中的aid特征（何时出现、交互类型等）

### 模型
LightGBM Ranker, 单模0.604，使用了9个不同超参训练的Lightgbm, 最终得分0.605

## 其他

1. 负样本采样：click 5%, carts: 25%, orders: 40%
2. cv策略：采用开源的方案，为了快速迭代，采用5%的数据进行训练，10%的数据作为验证
3. 消融实验，可以看到这个NN模型提升还是挺大的，提升了5个千分位

![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1676143052203-c7656281-db7b-4429-aa0d-3b4cf35de8ee.png#averageHue=%23fefefd&clientId=ucaaefed0-cb00-4&from=paste&height=600&id=u971916f7&name=image.png&originHeight=1199&originWidth=1627&originalType=binary&ratio=2&rotation=0&showTitle=false&size=142420&status=done&style=none&taskId=uc188f676-e414-4259-8788-f5d740c7e5b&title=&width=813.5)

# 2nd Place Solution Part 1

- 链接：[https://www.kaggle.com/competitions/otto-recommender-system/discussion/382839](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382839)
- 方案亮点：对于item cf的召回做了很多工作

## 召回阶段
由于这个比赛没有什么用户/商品特征，然后热门召回也不起作用，因此主要的召回方式包括：

1. session内交互过的aid
2. next action: 与共同访问矩阵类似，统计每个aid和下一个aid的出现次数，排序进行召回
3. itemcf，使用了多种矩阵和加权方式
   1. 矩阵：cart-order，click-cart&order，click-cart-order
   2. 加权方式：交互类别，共现时间距离，共现的顺序，session action 顺序，热门程度

## 排序阶段

### 特征工程

1. 聚合特征：session，aid, session*aid，以及分类别的计数
2. next action特征：候选商品与最后一个商品共同出现的次数
3. 时间：session和aid的开始时间和结束时间
4. itemcf分数，候选商品与session最后一个商品、最后一个小时商品、所有商品的最大、加和、加权和的itemcf分数
5. embedding相似度，word2vec生成的候选商品与最后一个商品的相似度，ProNE生成的session和aid相似度

### 模型
catboost ranker+lightgbm classifier融合

## 其他

1. 使用polars代替pandas，在数据merge时可以加速40倍（不过我自己用的时候很坑的一点是如果有空值会把数据类型改成float 64，虽然速度很快，但极大加剧了我的内存占用）
2. 使用TreeLite来加速lightgbm的推理速度(快2倍)，CatBoost-GPU比lightgbm-CPU的推理快30倍

# 2nd Place Solution Part 2

- 链接：[https://www.kaggle.com/competitions/otto-recommender-system/discussion/382790](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382790)
- 方案亮点：这是他们队另外一个人写的方案，写的有点少，item2item的特征做的比较出色

## 召回阶段
略，用的队友@psilogram的方案，没有写

## 排序阶段

### 特征工程
一部分略，用的队友@psilogram的方案，主要做了一些的item2item特征，包括：

1. 计数
2. 时间差
3. sequence difference(@psilogram发明的)
4. 2种以上提到的加权特征
5. 上述特征的聚合

最终得到400-500个特征

### 模型
xgboost+catboost

## 其他

1. 伪标签？：模型分两阶段，第一阶段训练好后输出oof prediciton, 然后作为特征再训练一次模型，虽然可能会过拟合，但是分数提升了
2. 使用了cudf和cuml进行加速

# 3rd Place Solution Part 1

- 链接：[https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013](https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013)
- 方案亮点：这个部分是Chris大神做的，非常简洁，没有各种花里胡哨的召回和特征工程，仅仅靠各种规则生成的共同访问矩阵就能单模0.601

## 召回阶段

1. session内交互过的aid
2. 共同访问矩阵，根据不同的规则一共做了20个共同访问矩阵，具体可以看他开源的notebook[https://www.kaggle.com/code/cdeotte/rules-only-model-achieves-lb-590/notebook](https://www.kaggle.com/code/cdeotte/rules-only-model-achieves-lb-590/notebook)

## 排序阶段

### 特征工程

1. session特征
2. aid特征
3. session aid交互特征
4. 共同访问矩阵生成的分数特征

### 模型
单模xgboost

## 其他

1. 使用了cudf进行加速，在4块v100的GPU上生成共同访问矩阵，生成了上百个，一个一分钟左右就能跑完，最终计算local cv挑了20个

# 3rd Place Solution Part 2

- 链接：[https://www.kaggle.com/competitions/otto-recommender-system/discussion/382975](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382975)
- 方案亮点：和上面的方案大差不差，主要是后面提到一两个trick比较有意思

## 召回阶段
用的Chris的候选，经过一点点调整

## 排序阶段

### 特征工程

1. 常规特征
2. item2item相似度特征：（w2v相似度，矩阵分解相似度，共同访问矩阵相似度），对session内的aid进行各种加权（时间、位置、类别）计算相似度，然后聚合（mean、max、sum等），如下图

![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1676192192198-34d4e62c-179e-47bb-bcb2-c6d1e61ce8ad.png#averageHue=%23fefcfb&clientId=ucaaefed0-cb00-4&from=paste&height=595&id=ucf5e6861&name=image.png&originHeight=892&originWidth=1736&originalType=binary&ratio=2&rotation=0&showTitle=false&size=241733&status=done&style=none&taskId=u3b116f93-4c21-41c2-8012-f76501c922c&title=&width=1157.3333333333333)

### 模型
xgboost

## 其他

1. 增加训练数据，由于主办方在训练和预测分割时，丢掉了一部分数据，于是他把这些数据也加进来训练，有0.0005 到 0.001的提升

![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1676192180483-c727ca56-b609-47c3-bfdd-967b1fb96a27.png#averageHue=%23fcfaf9&clientId=ucaaefed0-cb00-4&from=paste&height=353&id=u5dd99973&name=image.png&originHeight=529&originWidth=1917&originalType=binary&ratio=2&rotation=0&showTitle=false&size=108710&status=done&style=none&taskId=u999e678b-77b4-4c2d-9f03-33fe7287ad1&title=&width=1278)

2. 使用optuna来调整多折训练融合的权重
3. 代码已开源：[https://github.com/TheoViel/kaggle_otto_rs](https://github.com/TheoViel/kaggle_otto_rs)

另外，这里原本的第三名应该是另外一支队伍的，因为有个GM作弊，导致整只队伍被取消成绩，他们的方案也很值得学习，并且也有代码开源：[https://www.kaggle.com/competitions/otto-recommender-system/discussion/382879](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382879)

# 总结
这次比赛还是学到挺多的，更多的是认清了kaggle的水深，敬告大家组队时一定要擦亮双眼，防止队友作弊导致全队几个月的工作付之一炬，最后放一份自己的代码：<br />[https://github.com/lukan217/kaggle_otto_rec_sys](https://github.com/lukan217/kaggle_otto_rec_sys)

