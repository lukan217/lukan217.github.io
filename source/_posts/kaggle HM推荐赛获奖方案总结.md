---
title: Kaggle HM推荐赛获奖方案总结
tags:
- kaggle
- 推荐系统
categories:
- 比赛
---



<a name="uOrxS"></a>

# 1st place solution
文档链接：[1st place solution](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070)<br />整体框架：<br />![overview (1).png](https://cdn.nlark.com/yuque/0/2022/png/764062/1652519977145-990e767f-fd60-4ced-bd79-d24e225fd1a3.png)
<a name="agjGv"></a>



## 召回策略

1. 用户上次购买的商品
1. Item CF
1. 属于同个product code的商品
1. 热门商品
1. graph embedding：ProNE
1. 逻辑回归：训练Logistic回归模型，从前1000个热门商品中检索到50~200个商品

召回策略一共为每个用户召回了100-500个商品，使用HitNum@100来评估召回策略的质量，尽可能覆盖更多的正样本。
<a name="jbxxm"></a>



## 特征工程
| 类型 | 描述 |
| --- | --- |
| Count | user-item, user-category of last week/month/season/same week of last year/all, time weighted count… |
| Time | first,last days of tranactions… |
| Mean/Max/Min | aggregation of age,price,sales_channel_id… |
| Difference/Ratio | difference between age and mean age of who purchased item, ratio of one user's purchased item count and the item's count |
| Similarity |  item2item的协同过滤分数, item2item(word2vec)的余弦相似度, user2item(ProNE)的余弦相似度 |

<a name="WKsy0"></a>



## 排序模型
5个lightgbm classifier  + 7个catboost classifier<br />不同的分类器分别用不同的时间跨度以及召回数量的数据进行训练，如下图所示：<br />![cv.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1652512638734-0abf8e61-2a80-4042-9547-38db5e32de91.png)
<a name="rrbKI"></a>



## CV策略
使用最后一周作为验证集
<a name="rZbAm"></a>



## 优化技巧

1. 模型推理优化：使用TreeLite 来优化lightgbm推理的速度（快了2倍），caboost-gpu版本比lightgbm-cpu版本快了30倍
1. 内存优化：类别特征用了labelencoder，并且使用了reduce_mem_usage函数
1. 特征存储：将创建的特征保存为feather格式，方便使用
1. 并行：将用户分为28组，在多个服务器上同时进行推理
1. 机器：128g内存，64核CPU, TITAN RTX GPU（真特么有钱啊！）
<a name="WQeOZ"></a>



## 亮点

1. 使用逻辑回归作为pre-ranker进行召回

1. 模型融合，用了12个模型进行融合

1. 各种优化技巧
    <a name="Q561R"></a>


# 2nd place solution
文档链接：[2nd place solution](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324197)
<a name="tfTnq"></a>



## 召回策略

1. 热门商品，基于不同的维度进行召回：
   1. 用户属性：如不同年龄、地域购买的热门商品（其中，基于地域postal_code的召回最为显著的提升了分数）
   1. 商品属性：如果用户购买了具有特定属性的商品，会寻找具有相同属性的热门商品
   1. 使用不同的时间窗口：1、3、7、30、90天。
   
2. 用户历史购买过的全部商品

3. 使用MobileNet embedding计算的图像相似度及进行召回

4. graph embedding：random walk
    <a name="xCAkG"></a>


## 特征工程

1. 用户基本特征：包括购买数量、价格、sales_channel_id

1. 商品基本特征：根据商品的每个属性进行统计，包括：times, price, age, sales_channel_id, FN, Active, club_member_status, fashion_news_frequency, last purchase time, average purchase interval

1. 用户商品组合特征：基于商品每个属性的统计信息，包括：num, time, sales_channel_id, last purchase time, and average purchase interval

1. 年龄商品组合特征：每个年龄组的商品受欢迎程度。

1. 用户商品回购特征：用户是否会回购商品以及商品是否会被回购

1. 高阶组合特征：例如，预测用户下次购买商品的时间

1. 相似度特征：通过各种手段计算商品与客户购买的商品的平均相似度
    <a name="M5TgH"></a>


## 排序模型
魔改的lightgbm ranker，使用lambdarankmap作为目标函数（从xgboost里面copy的代码），比lightgbm原生的lambdarank目标要好。
<a name="J9f3W"></a>



## cv策略
2-3个月作为训练集，最后一周作为测试集。
<a name="GX3mX"></a>



## 亮点

1. 使用MobileNet生成了图像的embedding特征

1. 各种高阶特征：用户是否会回购商品以及商品是否会被回购、用户下次购买商品的时间等

1. 使用lambdarankmap作为目标函数的lgb ranker
    <a name="G7QZT"></a>


# 3rd place solution
文档链接：[3rd place solution](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129)<br />作者说他整体建模的套路都和其他人差不多，因此分享了两点他觉得跟别人不同但是有用的东西：

1. 召回策略特征：这个商品被哪种策略召回，以及被召回时的排名，加上这两个特征，以及将召回的数量从几十增加到上百，将LB分数从0.02855提高到0.03262，直接从银牌区进去了金牌区，此外，如果只增加召回的数量，而不添加增加召回的特征的话，CV分数非常低。

1. BPR  矩阵分解得到的用户-商品相似度特征：用户和商品的相似性特征对排序模型很重要。关于item2item相似性的特征，例如Buyd together计数和word2vec，在大多数竞争对手的模型中都很常用，这些特征也大大提高了我的分数，但最能改善我的模型的是通过BPR矩阵分解获得的user2item相似性。这个BPR模型是在目标周之前（每周训练一个BPR）使用[implicit](https://implicit.readthedocs.io/en/latest/bpr.html)训练所有交易数据的。BPR相似性的auc约为0.720，而整个排序模型的auc约为0.806，其他单一特征的最佳auc约为0.680。最后，这个相似性特征将我的LB分数从0.03363提高到了0.03510，这将我从金牌区带到了奖品区。
    <a name="fU9mA"></a>


# 4th place solution

文档链接：[4th place solution](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324094)<br />整体思路：<br />![arch.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1652517925129-614ba42d-e10b-44aa-87ff-980f75d216a0.png)
<a name="qL1xq"></a>



## 召回策略

1. item CF
1. 最近购买：用户最近购买的12个商品
1. 热门商品：上周的热门商品
1. Two Tower MMoE：作者主要关注这个模型的训练，因为它不仅能够为用户生成任意多的候选商品，还能够创建用户-商品相似度特征，这是模型的架构：

![loss.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1652518291522-78e85301-578d-4bbf-9d28-f7b84a34311a.png)<br />![item-tower.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1652518252440-b3e414e8-19fc-4429-ad96-38c68c168fb7.png)<br />![user-tower.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1652518271421-f86f7bb6-b8d5-409a-9a52-ebd109add659.png)<br />对于用户侧的tower，作者还用了一个门控网络，以确保用户塔可以通过使用不同的expert为最近的活跃客户和非活跃客户进行学习。
<a name="ZYVEp"></a>



## 冷启动

- 用户冷启动：双塔MMoE可以使用用户基础特征为没有购买日志的用户生成候选商品。

- 商品冷启动：除了商品的基本特征，作者还用了图像和文本的特征
   - 文本：从商品描述中提取TF-IDF特征，使用SVD+K-Means对商品进行聚类。然后使用聚类的label作为特征。
   
   - 图像：使用预训练的tf_efficientnet_b3_ns提取图像向量，使用PCA+K-均值聚类。然后使用聚类的label作为特征。
     <a name="eXWZi"></a>
   
     
## 亮点

1. 召回策略用了Two Tower MMoE

1. 文本和图像特征用的是embedding过后再进行一次聚类的标签
    <a name="BctOx"></a>

  

# 5th place solution

文档链接：[5th place solution](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324098)
<a name="yjDMN"></a>



## 召回策略

1. 用户上一篮子的购买（以及和用户上次购买商品相同product code的商品），最近购买的商品
1. User CF
1. Item CF
1. word2vec
1. 经常会被一起购买的商品
1. 根据用户的属性（年龄、性别等）召回的热门商品

关于召回的数量，作者使用一个正样本的比例作为阈值来控制的，比如把阈值设为0.05的话，然后某个策略召回100个商品，正样本比例为0.01，那么可以召回少一点的商品，比如50个，这样刚好可以使得正样本比例刚好卡在0.05.
<a name="EANNj"></a>



## Embedding方法

1. 商品的图像信息：使用swin transformer来提取embedding

1. 商品的文本信息：使用SentenceTransformer提取

1. tf-idf获取商品的embedding

1. word2vec获取商品的embedding
    <a name="V67L5"></a>


## 特征工程

1. 用户特征
1. 商品特征
1. 用户-商品特征：如相似度特征，聚合的统计特征等

对于第1部分和第2部分，可以进行计算并保存一次，然后与第3部分的特征合并。这种方法可以节省很多时间，尤其是在推理的时候。
<a name="iBcjc"></a>



## 模型
作者用了大量时间在召回策略的设计上，因此只用了lightgbm单模
<a name="pKWxi"></a>



## 亮点

1. 召回策略非常丰富，一共用了21种
1. 图像和文本的embedding，图像用了swin transformer，文本用了SentenceTransformer

