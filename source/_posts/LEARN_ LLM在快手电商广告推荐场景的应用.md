---
title: LEARN_ LLM在快手电商广告推荐场景的应用
mathjax: true
date: 2024/6/15
tags:
- 推荐系统
- 大语言模型
- 论文阅读
categories: 
- 推荐系统
---

今天继续分享一篇大模型在推荐系统中的落地应用工作，是快手今年5月份发表的论文《Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application》。<br />太长不看版：<br />这篇文章主要做了两个工作：

- 工作1：使用冻结的LLM提取文本embedding后，设计了一个基于transformer的双塔结构，用对比学习的方式训练用户行为序列数据，以提取更适用于推荐任务的user embedding和item embedding，
- 工作2：在排序模型中加一个CVR辅助任务，使用1学习到user embedding和item embedding接入一个MLP网络和推荐目标做进一步对齐，然后取出中间层特征供排序模型使用

方法其实和上次介绍的小红书的NoteLLM有点类似的地方，都是想用大模型作为特征提取器来提取item文本中的语义信息来弥补推荐模型中的冷启动以及长尾物品由于行为稀疏学不好的问题，并且都使用对比学习的范式进行学习，不过小红书的采用i2i的方式来训练，产出一个item embedding，最终用在i2i的召回上，这篇用的是u2i的方式，能够同时产生user embedding和item embedding，看文章的线上实验最终两个embedding是用在排序模型中作为特征使用（对应工作2），并没有用于u2i召回。<br />所以个人认为这篇文章对于实践的价值参考意义更大的可能是工作2，虽然笔墨很少，但毕竟是有上线的，工作1花里胡哨一通操作，最终也就离线自己搞了个数据集和其他几个方法跑了跑对比，为了发文章也可以理解，但是思想也可以稍微参考下吧。
<a name="UCXG4"></a>

# 背景
现有的推荐系统模型都是通过ID embedding学习用户和物品之间的交互来表示用户和物品，然而这种方式忽略了，物品文本描述中包含的语义信息，同时，对于一些行为数据少的用户和物品（冷启动和长尾）ID embedding是学不好的，但是LLM对于语义信息的表征能力是很强的，所以自然就会想用LLM学习物品描述中的语义信息来改善推荐中的冷启动和长尾问题。<br />现有的用LLM来做推荐的大多是通过构建prompt将推荐的数据文本化作为LLM的输入，然后通过生成式的方式来推荐物品，但受限于计算性能以及LLM的输入长度，所以也只能在几个玩具数据集上跑一跑，文章把这种方式称为Rec-to-LLM, 而文章要做的工作就是LLM-to-Rec，将LLM用作特征提取器，将推荐任务作为训练目标，不仅有利于从 LLM 的开放世界领域无缝过渡到RS的协作领域，还能确保更好地满足工业在线 RS 的实际需求。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/764062/1718162974652-57d48fce-0545-4cd1-8d26-a81fbed28d4d.png#averageHue=%23f7efe7&clientId=u60cadb94-7a22-4&from=paste&height=293&id=udca5cca6&originHeight=586&originWidth=1201&originalType=binary&ratio=2&rotation=0&showTitle=false&size=140639&status=done&style=none&taskId=ue5be2db7-d2a8-4e38-a426-5c65ddcb068&title=&width=600.5)
<a name="t2SLs"></a>

# 方法
提出方法的框架叫做Llm-driven knowlEdge Adaptive RecommeNdation，简称LERAN，采用的是双塔的架构来进行自监督学习，用户塔和物品塔均由内容嵌入生成模块（CEG）和偏好理解模块（PCH）组成。模型学习的目标是预测感兴趣的下一个物品，也就是序列推荐，所以训练时会抽出用户的历史行为序列，然后在中间截断一下分成两个序列，$U_i^{\text {hist }}=\left\{\text { Item }_{i, 1}, \text { Item }_{i, 2}, \ldots, \text { Item }_{i, H}\right\}$, $U_i^{\text {tar}}=\left\{\text { Item }_{i, H+1}, \text { Item }_{i, H+2}, \ldots, \text { Item }_{i, H+T}\right\}$，$U_i^{\text {hist }}$作为用户塔的输入，$U_i^{\text {tar}}$作为物品塔的输入。<br />内容嵌入生成CEG 模块采用了预训练的LLM（文章用的Baichuan2-7B）作为物品编码器。编码的内容包括：标题、类别、品牌、价格、关键词和属性，这里LLM是冻结参数的，防止灾难性遗忘问题，最终会提取出每个token最后一层的隐向量做平均池化后作为物品最终的表征。<br />偏好理解PCH模块是为了将LLM生成的内容embedding与推荐任务进行对齐，弥补的开放世界知识与协作知识之间的领域差距，采用推荐任务的自监督训练目标来指导模型优化。PCH 模块使用用户交互过的物品的内容嵌入序列作为输入，过一个使用因果注意力的transformer编码器，最终生成user embedding或者item embedding，下面画的这个图右侧应该是用户塔的，很好理解，但是最终生成的user embedding没说是去平均池化还是最后一个token。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/764062/1718365039868-9b4bef2a-d97d-44e8-9c92-c96eb716d2ce.png#averageHue=%23f8f3ef&clientId=u80a76b04-4b88-4&from=paste&height=342&id=ga1OC&originHeight=684&originWidth=1235&originalType=binary&ratio=2&rotation=0&showTitle=false&size=151117&status=done&style=none&taskId=ua4e728f9-686d-46ca-a17f-1f2f367e11a&title=&width=617.5)<br />物品塔就写的有点难以理解了，文章说设计了三个变体，最终采用的是变体1，其他两个效果不好，来具体看下三个变体怎么搞的：

- 变体1：采用和用户塔一样的结构和权重，为每个目标物品进行编码
- 变体2：文章说用了自注意力，应该是把变体1的因果注意力换成了自注意力?那为什么输入的时候要换成单个物品输入，而不是用序列输入？这里有点不太理解
- 变体3：这个就是直接把LLM生成的内容嵌入拿来当成item embedding，连加一个映射层都不做了

![image.png](https://cdn.nlark.com/yuque/0/2024/png/764062/1718365031906-fe9898ea-f84e-4819-976a-e36350ff0f78.png#averageHue=%234ebdc9&clientId=u80a76b04-4b88-4&from=paste&height=337&id=OxeCh&originHeight=673&originWidth=2186&originalType=binary&ratio=2&rotation=0&showTitle=false&size=212503&status=done&style=none&taskId=u29221e3b-a780-46b2-9134-73a126fa1f4&title=&width=1093)<br />有个不太理解的地方就是这个物品塔为啥用序列作为输入，因为文章说训练时是用一个物品序列作为输入，但是在推理时，只用单个物品，这样不会存在训练和测试不一致的情况吗？<br />另外还有一个训练成本的问题，因为这里又加了一个transformer的encoder，那么用的序列长度应该也不能很长，而且也不知道这个transformer到底能起到多大作用，毕竟行为序列里面噪声应该还是很多的，所以我觉得更简单粗暴的一个方法是用户塔序列pooling后接一个MLP，这样可以支持更长的序列，物品塔在拿到LLM的表征后也接一个MLP，这样的方式不知道会不会更好。<br />不管咋样，就当他通过某种手段拿到了LLM提取得到的user embedding和item embedding，然后就是进行对比学习了，正样本用当前用户交互的物品，负样本用其他用户交互的物品，损失函数用的也是InfoNCE<br />$\mathcal{L}=-\sum_{i=1}^{N_u}\sum_{j=1}^H\sum_{k=1}^T\log\frac{e^{s(E_{i,j}^{user},E_{i,k}^{item})}}{e^{s(E_{i,j}^{user},E_{i,k}^{item})}+\sum_{z\neq i}\sum_ke^{s(E_{i,j}^{user},E_{z,k}^{item})}}$
<a name="ncIFs"></a>

# 实验
<a name="SsUxJ"></a>

## 离线实验
离线实验的结果没啥好看的，主要就是自己造个数据集然后和几种方法比了比再做个消融，直接说结论：

1. LLM embedding比传统的ID embedding以及bert生成的更有优势

![image.png](https://cdn.nlark.com/yuque/0/2024/png/764062/1718365484772-cc168940-f0c4-445d-b003-737a6dac8c66.png#averageHue=%23eeeeed&clientId=u80a76b04-4b88-4&from=paste&height=128&id=u03d70a02&originHeight=256&originWidth=992&originalType=binary&ratio=2&rotation=0&showTitle=false&size=47256&status=done&style=none&taskId=ud2132b8b-850c-49dd-b2c9-bb7ea7dab1a&title=&width=496)

2. 和现有的Rec-to-LLM比也更有优势，在公开数据集MovieLens也比现有的SOTA: HSTU和SASRec更有优势

![image.png](https://cdn.nlark.com/yuque/0/2024/png/764062/1718365649439-795020a9-078e-4656-b03c-47de74034b49.png#averageHue=%23efefee&clientId=u80a76b04-4b88-4&from=paste&height=124&id=u24bdbb19&originHeight=248&originWidth=1088&originalType=binary&ratio=2&rotation=0&showTitle=false&size=58270&status=done&style=none&taskId=uf0ff0396-51f9-42bc-9e31-4f7423542b3&title=&width=544)<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/764062/1718365700356-a3535d75-63bb-4a47-8733-2c4408ce40ff.png#averageHue=%23f0efef&clientId=u80a76b04-4b88-4&from=paste&height=225&id=u356ad9fe&originHeight=450&originWidth=1093&originalType=binary&ratio=2&rotation=0&showTitle=false&size=114011&status=done&style=none&taskId=u04e2e090-82b3-4a3e-afa0-a5f45f5b985&title=&width=546.5)

3. 直接冻结LLM提取特征再接transfomer比用lora微调更有优势

![image.png](https://cdn.nlark.com/yuque/0/2024/png/764062/1718365855259-9478452d-1fed-4837-9757-a7d11251b8a2.png#averageHue=%23f2f2f1&clientId=u80a76b04-4b88-4&from=paste&height=177&id=ub66fe992&originHeight=353&originWidth=1104&originalType=binary&ratio=2&rotation=0&showTitle=false&size=65077&status=done&style=none&taskId=uee226286-c4d1-4ca7-b395-0e75d761747&title=&width=552)
<a name="Ypodz"></a>

## 在线实验
在线实验这里就只提到了怎么去把所提出方法的user embedding和item embedding用在排序模型中，所以模型提出来的这个双塔肯定是不能用在u2i召回的，想想成本也很高。<br />用的具体方法如下，首先在排序模型旁边添加一个辅助的CVR任务，学习目标就是转化率，输入是LEARN学习得到的user emb和item emb，concat起来后过一个MLP，再取输出层前的中间向量mid emb和user emb以及item emb作为正常的特征和其他特征拼接起来喂给排序模型，这里做的事情就是把user emb和item emb再一次和推荐任务的目标进行强制对齐操作，这时候mid emb里面不仅包含了语义信息，也学习到了一些推荐的信号，所以作为一个特征加入到排序中作用应该挺大的<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/764062/1718126696275-c2f518fd-6497-43db-b235-f0a4ef05d4c2.png#averageHue=%23fafafa&clientId=ua2714442-4f05-4&from=paste&height=284&id=u86d16286&originHeight=568&originWidth=1299&originalType=binary&ratio=2&rotation=0&showTitle=false&size=72879&status=done&style=none&taskId=ua98f8594-a2b9-4fbd-a9df-d4a29df9ded&title=&width=649.5)<br />看效果，离线训练的AUC涨幅明显：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/764062/1718127001459-688c4444-0ced-48ac-b0b6-e15c00f45506.png#averageHue=%23f0efee&clientId=ua2714442-4f05-4&from=paste&height=184&id=u00c04b35&originHeight=368&originWidth=1111&originalType=binary&ratio=2&rotation=0&showTitle=false&size=56543&status=done&style=none&taskId=u486bf0a2-a200-41f1-a1ec-fbd6adf461b&title=&width=555.5)<br />线上看利润和AUC指标也有所提升，拆分了下指标涨幅主要来源于长尾和冷启的用户/商品，借助LLM的开放世界知识可以有效改善冷启动，看起来是比较符合预期的，但是比较神奇的一点是这里的AUC涨幅跟上面的离线结果对应不太上啊<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/764062/1718127015000-64c8ceda-6a3a-4f08-a955-42ab7e1903cc.png#averageHue=%23f0efee&clientId=ua2714442-4f05-4&from=paste&height=360&id=K231x&originHeight=720&originWidth=1356&originalType=binary&ratio=2&rotation=0&showTitle=false&size=145228&status=done&style=none&taskId=ufa29e15e-26d0-4547-a1ba-51ae72e5031&title=&width=678)<br />根据我自己的理解，这里这个能起到作用的原因有两个：

1. LLM包含的丰富的语义信息为现有模型提供了信息增量，对行为数据少的user和item较为友好
2. 两次对齐操作保证了LLM特征的有效性，首先user emb 和item emb在上游LEARN框架中和推荐任务的目标通过对比学习进行了对齐，而在下游排序模型中，又通过CVR任务又强制把user emb和item emb和推荐任务的目标再进行了一次对齐，所以这里不管是user emb、item emb，还是取到的中间层特征向量，都不仅包含了原始大模型里面的丰富的语义信息，还包含了推荐目标的一些信息，可以作为推荐的特征直接加入到排序模型中使用

以往在做推荐的时候，有个往往起不到作用的操作是都想把bert、resnet等模型产出的几十维的多模态向量特征加直接加入到精排模型中，首先不提这些特征包含的增量信息到底有多少，还有一个关键原因应该是这些特征本质上都是在其对应的预训练任务上做的，与推荐任务的目标是不一致的，直接加入到模型中可能会被当成噪音，所以需要做个改造，而这篇文章做的两个工作都是为了对这些特征做改造，使其与推荐任务进行对齐，最后再作为特征使用。
<a name="Z6RGh"></a>

# 参考
[Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988)
