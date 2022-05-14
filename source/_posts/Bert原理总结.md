---
title: BERT原理总结
mathjax: true
tags:
- 自然语言处理
- 深度学习
categories:
- 深度学习
---



最近在做nlp相关的任务，发现无脑上bert就能达到很好的效果了，于是就去看了原论文，写篇文章好好总结一下吧！
<a name="YRyi1"></a>



# 背景
在计算机视觉领域，预训练已经被证明是行之有效的了，比如ImageNet，训练了一个很大的模型，用来分类1000种东西，然后底层的模型架构就能很好的捕捉到图像的信息了，就可以直接迁移到其他任务上，比如一个猫狗的二分类问题，就只需要把模型拿来微调，接一个softmax输出层，然后重新训练几个epoch就能达到很好的效果了。类似的预训练一个大模型然后拿来做迁移学习的思想也被用在了nlp上，语言模型的预训练用在下游任务的策略主要有两种：

1. 基于特征（feature-base）：也就是词向量，预训练模型训练好后输出的词向量直接应用在下游模型中。如ELMo，用了一个双向的LSTM，一个负责用前几个词预测下一个词，另一个相反，用后面几个词来预测前一个词，一个从左看到右，一个从右看到左，能够很好地捕捉到上下文的信息，不过只能输出一个词向量，需要针对不同的下游任务构建新的模型。
1. 基于微调（fine-tuning）：先以自监督的形式预训练好一个很大的模型，然后根据下游任务的不同接一个输出层就行了，不需要再重新去设计模型架构，如OpenAI-GPT，但是GPT用的是一个单向的transformer，训练时用前面几个词来预测后面一个词，只能从左往右看，不能够很好的捕捉到上下文的信息。

ELMo虽然用了两个单向的LSTM来构成一个双向的架构，能够捕捉到上下文信息，但是只能输出词向量，下游任务的模型还是要自己重新构建，而GPT虽然是基于微调，直接接个输出层就能用了，但是是单向的模型，只能基于上文预测下文，没有办法很好的捕捉到整个句子的信息。<br />因此，BERT（Bidirectional Encoder Representations from Transformers）就把这两个模型的思想融合了起来，首先，他用的是基于微调的策略，在下游有监督任务里面只需要换个输出层就行，其次，他在训练的时候用了一个transformer的encoder来基于双向的上下文来表示词元，下图展示了ELMo、GPT和BERT的区别：<br />![](https://cdn.nlark.com/yuque/0/2022/svg/764062/1649144748837-d0dd42fe-0a8b-4e29-820f-c9b923850db3.svg#clientId=u0872f9ad-41c6-4&crop=0&crop=0&crop=1&crop=1&from=paste&id=u00a83598&margin=%5Bobject%20Object%5D&originHeight=392&originWidth=611&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=uf43e63ab-467c-46cb-ad61-51e39d67573&title=)<br />BERT很好的融合了ELMo和GPT的优点，论文中提到在11种自然语言处理任务中（文本分类、自然语言推断、问答、文本标记）都取得了SOTA的成绩。
<a name="HEjdw"></a>



# 核心思想

BERT的模型结构采用的是transformer的编码器，模型结构如下，其实就是输入一个$n\times h$（n为最大句子长度，h为隐藏层的个数）的向量，经过内部的一些操作，也输出一个$n\times h$的向量。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1649397385919-e4ce313b-f3d7-4dee-8c05-a1ee3e48c5f3.png)<br />根据模型的一些参数设置的不同，BERT又分为：

- $BERT_{BASE}$：transformer层12，隐藏层大小768，多头注意力个数12，总共1.1亿参数
- $BERT_{LARGE}$：transformer层24，隐藏层大小1024，多头注意力个数16，总共3.4亿参数

BERT主要的工作在于对于**输入表示的改造**以及**训练目标的设计**。
<a name="CaisL"></a>



## 输入表示
在自然语言处理中，有的任务的输入可能只需要一个句子，比如情感分析，但是有的任务的输入是需要一对句子的，比如自然语言推断，因此，为了使Bert能够用在更多的下游任务上，BERT的输入被设计为不仅可以输入一个句子，也可以输入一个句子对。<br />不管输入的是一个句子还是句子对，BERT的输入的第一个词元都是一个特殊的词元<CLS>，作为句子的开始，并且这个<CLS>在最后输出的表征中也有很重要的作用，对于两个句子，BERT用一个分隔符<SEP>，因此：

- 对于一个句子，BERT的输入结构为：<CLS>句子<SEP>
- 对于一个句子对，BERT的输入为：<CLS>句子1<SEQ>句子2<SEP>

由于注意力机制是无法捕捉到位置信息的，因此BERT还加了一个position embedding，这里的position embedding的参数是自己学出来的，用来加在每个词元上的token embedding。<br />并且，为了区分句子对，BERT又训练了一个两个Segment Embeddings，分别加在原来的两个句子对应的token embedding上。<br />因此，最后BERT的输入就是三个embedding相加的结果，如下图所示：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1649399414503-4c3e9e2a-5555-4d35-937b-d6f4eee065f2.png)
<a name="w8GyY"></a>



## **Masked Language Model (MLM)**
前面说到，之前的预训练模型都是单向的，也就是用前几个词来预测下一个词，这样有个缺陷就是无法捕捉整个句子的上下文信息。因此BERT采用了在输入时随机mask词元的方式，然后基于上下文，在输出层里面预测这些被mask的词元，其实这就是完型填空了，就像我们以前高中英语做的一样，要能够填空，那么就得对上下文的语义有一个比较深入的了解，因此bert最后训练出来的参数就能够很有效的表征整个句子的语义。<br />具体来说，输入的时候会会把一个句子中的词随机mask掉一部分，比如：“你笑起来真好看”变成“你<mask>起来真<mask>看”，然后还会记住这些被mask住的词的位置，然后再输出的地方找到这些词元的对应的表征，再接一个和词典大小一样的输出层，就可以预测这些位置上被<mask>掉的词是什么了，训练时使用的损失函数也使用交叉熵。<br />但是该遮掉多少词也是个问题，论文里给了一个15%的比例，在训练时将15%的词替换为用一个特殊的“<mask>”替换，不过在训练时可以这么做，在我们微调的时候可就没有<mask>词元了，因此BERT选择这样的设计：

- 80%时间为特殊的“<mask>“词元（例如，“this movie is great”变为“this movie is<mask>”；
- 10%时间为随机词元（例如，“this movie is great”变为“this movie is drink”），这里的目的是为了引入一些噪声，有点像纠错了；
- 10%时间内为不变的标签词元（例如，“this movie is great”变为“this movie is great”）
<a name="DFglo"></a>



## **Next Sentence Prediction (NSP)**
因为研究者想让bert还能够适应像自然语言推理这类的任务，因此还加入了另一个任务，也就是当输入的是一个句子对的时候，BERT会预测这两个句子在上下文中是否是相邻的，具体在训练时，就会有50%概率输入的句子对是相邻的，50概率输入的句子对是不相邻的，其实就是一个二分类任务，这里刚好用之前提到的句子开头那个<CLS>标记最终输出的隐藏层再接一个softmax二分类输出层就行了，然后用交叉熵来作为损失函数。<br />最终把MLM的损失函数和NSP的损失函数加起来就是BERT最终的损失了，可以用Adam来做优化。
<a name="y0agw"></a>



# BERT的使用
接下来主要讲讲BERT在各个任务上是怎么使用的，其实也就是接一个输出层啦。

1. 文本分类任务：和NSP类似，在<CLS>这个词元的输入顶部接一个softmax分类层
1. 问答任务：输入一个文本序列，需要从这个序列中找到答案的位置，就是接两个输出层，一个用来代表答案开始的地方，一个用来代表答案结束的地方。
1. 命名实体识别（NER）任务：输入一个文本，标记文本中每个词元属于什么类型，直接把每个词元的输出向量输入到一个分类层就行。

具体在使用的时候，直接使用huggingface的[🤗 Transformers](https://huggingface.co/docs/transformers/index)就行，里面内置了很多预训练模型，并且对于每个任务也都有很好的封装，使用成本很低。
<a name="EwtoE"></a>



# 参考

1. [[1810.04805] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
1. [BERT Explained: State of the art language model for NLP | by Rani Horev | Towards Data Science](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
1. [14.8. 来自Transformers的双向编码器表示（BERT） — 动手学深度学习 2.0.0-beta0 documentation](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/bert.html)
