---
title: Transformer原理： Attention is All you need
mathjax: true
tags:
- 面经
categories:
- 深度学习
---
transformer自诞生以来，基本上在每个领域都取得了非常大的成功，如nlp领域的Bert、GPT，cv领域的ViT，swin transformer，推荐系统领域的autoint，behavior sequence transformer，还有时序里面的tft、informer，以及强化学习也搞了个Decision Transformer，而这些都源自于谷歌团队在2017年提出的这篇文章《Attention is All you Need》，本着阅读经典，顺便复习面经的精神，这次我们就来阅读transformer这篇论文，深入到每一个细节之中，确保对这个模型知根知底，当然，具体在写的时候不会严格按照原文来，而是按照我自己的想法来进行组织的。

# 背景
在传统的序列建模任务（如语言模型，机器翻译）中，一般使用的模型架构都是循环神经网络（LSTM和GRU），并且都是一个encoder-decoder的架构。这种基于RNN的模型结构不管在输入或者输出一个序列的时候都是把当前隐状态$h_t$建模成一个关于当前输入以及上一时刻隐状态的函数，即$h_t = f(h_{t-1},X_t)$，这种自回归式的建模方法意味着他只能串行计算，而没办法并行处理，如果序列的长度很长的话，计算就会很慢，除了通过把batch_size增大来提高运算速度之外好像也没别的方法，并且这么做对于内存要求还比较高。由于在序列这个维度上只能进行串行计算，这也成了模型计算速度的瓶颈所在。<br />有一些工作想要突破RNN这个无法并行的问题，比如Extended Neural GPU，ByteNet，ConvS2S等，但是这些网络都是用CNN作为模型的一部分，因为CNN是可以实现并行计算的，但是在长序列问题上还是存在问题，CNN很难捕捉序列上两个离得很远的点的依赖关系。<br />注意力机制是可以实现并行的，而且他对于远距离的两个点的依赖关系建模的也比较好，也被运用在了nlp的各种任务中，但是更多的是和RNN进行结合使用，增强RNN的效果，起到锦上添花的作用，还是突破不了RNN的局限性。<br />因此，这篇文章提出的Transformer就是想要用一个纯粹的注意力机制来解决机器翻译问题，当然也是采用encoder-decoder的架构，不过encoder和decoder都是基于自注意力，这么做的优点有以下三个：

1. 长序列建模，可以捕捉长序列之间的依赖关系
2. 可以并行计算, 在工业界应用比较友好
3. 效果好，在一系列任务上吊打其他模型

# 模型结构
基本上所有的序列建模模型都是采用encoder-decoder的架构，encoder负责把输入的序列表征$(x_1,...,x_n)$编码成另一个序列$(z_1,...,z_n)$，然后decoder再把编码好的$(z_1,...,z_n)$解码成输出$(y_1,...,y_n)$, 但是编码器和解码器的具体实现方式不同，以RNN系列的模型举例，都是在每个时间步$t$上都采用自回归的方式，把当前时间步的输入分为两个，一个是当前时间步的输入以及上一个时间步的hidden state，如对于编码器$z_t = f_{encoder}(z_{t-1},x_t)$, 而对于解码器$y_t = f_{decoder}(y_{t-1},z_t)$。<br />这里的transformer整体上也是采用同样的encoder-decoder架构，不过编码器和解码器的函数换成了纯注意力机制。来看一下他整体的架构，整体的结构还是encoder+decoder的方式，encoder接收来自一个句子的每个词embedding，为了表征每一个词的位置信息，先把句子的每个词的embedding加上一个位置编码（positional embedding），这是因为transformer的自注意力机制计算时不像RNN那样有先后顺序，对所有词向量都是一视同仁的，而decoder这边接收的则是要翻译的目标句子的词embedding，同样也加上位置编码，同时也接受来自encoder的输入，最后用softmax输出每一个位置上每个词元可能的概率。接下来再说一下encoder和decoder的一些细节。

![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1682947966354-94b238d6-2831-40e1-8261-d089d31f2ea9.png#averageHue=%23e8d8b1&clientId=ucb8551b1-5e38-4&from=paste&height=665&id=bSmYE&originHeight=1329&originWidth=973&originalType=binary&ratio=2&rotation=0&showTitle=false&size=158965&status=done&style=none&taskId=u46468e17-f1e6-4414-9043-8d82622cb05&title=&width=486.5)

## Encoder
首先看encoder这边，encoder由6个相同的层组成，每个层都有两个子层，第一个子层是多头注意力层，第二个子层是一个基于位置的前馈神经网络层，这两个子层之间使用了残差连接和layer normalization，用公式来说明的话就是，对每个子层的输出做了这样一个操作：<br />$LayerNorm(x+Sublayer(x))\\$<br />这边的$x$就是子层的输入，$Sublayer(x)$就是子层的输出，把输入和输出加起来，就是一个残差连接，然后再使用LayerNorm对输出进行层归一化。

## Decoder
再来看decoder这边，decoder同样由6个相同的层组成，每个层由三个子层组成，其中，两个子层和encoder的结构类似，多头注意力层和基于位置的前馈神经网络层，但是这个多头注意力层采用了mask的方式，这里的mask是指把当前词元之后的词元mask掉，不参与注意力的计算，这是因为对于翻译任务来说，训练时你能知道完整目标句子的输入，但是在预测时词元只能一个个生成，没办法看到后面的词，所以需要在训练时也把后面的词也给屏蔽掉。然后decoder在这两个子层之间又插入了一个子层，用来接收encoder的输入做注意力的计算，这个子层也是一个多头注意力层，细节之后展开。

## 注意力机制
首先说一下注意力机制的一些基本概念，注意力机制其实就是一个加权函数，要加权的东西，我们把它称为Value，既然是加权，权重如何计算呢？在注意力机制里面，我们一般是通过计算Query和Key的相似度得到的权重，每个Key和Value都是一一对应的，假设有n个key和value对，我们就可以通过一个query分别计算和key的相似度，得到n个相似度，这个就可以当作权重，然后乘到value里面，就可以得到加权后的输出。<br />这里的Query、Key、Value也就是注意力机制的三个要素，俗称QKV，**一句话概括注意力机制就是使用Q和K计算相似度作为权重来对V进行加权**，根据不同的相似度计算方法我们就有不同的注意力函数，transformer用的是缩放点积注意力。

### 缩放点积注意力
衡量向量相似度的一个方式就是计算他们的点积，因此点积便可以作为一种注意力函数，transformer使用的缩放点积注意力公式如下：<br />$\operatorname{Atention}(Q,K,V)=\operatorname{softmax}(\dfrac{QK^T}{\sqrt{d_k}})V$<br />这里的Q和K和V都是一个矩阵，Q之所以是个矩阵是因为transformer中输出都是多个位置的，每个Query对应一个位置，所以直接用矩阵的方式计算便可以并行计算，加快效率，这也是transformer的优势所在。<br />对输出的相似度使用了softmax可以把每个query下的相似度归一化，加起来正好是1。<br />这里点积还进行了一个缩放操作，即除以$\sqrt{d_k}$, 为什么要进行这样一个操作呢？具体来说，如果我们仅仅做点积操作，当向量的维度$d_k$很大时，点积的结果也会变大。因为点积操作本身就是将两个向量的对应元素相乘后再求和，如果向量的维度增大，点积的结果会相应地增大。这会导致点积注意力计算softmax时，输入值过大可能会导致梯度消失问题。因为softmax函数的输出是一个概率分布，而其梯度在其输入值非常大或非常小的时候会变得非常小。这种情况下，在反向传播中梯度就会消失，影响模型的学习。为了避免这个问题，我们需要对点积的结果进行缩放，即除以$\sqrt{d_{k}}$。这样做的主要目的是使得点积的结果的范围不会随着d_k的增大而变得过大，从而避免梯度消失的问题，使得模型能够更好地学习和优化。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1683021721901-d303b0e5-a809-4a92-ab0f-6ad41b23c984.png#averageHue=%23e8e8e7&clientId=ucb8551b1-5e38-4&from=paste&height=260&id=u101479f6&originHeight=520&originWidth=396&originalType=binary&ratio=2&rotation=0&showTitle=false&size=27728&status=done&style=none&taskId=u1132fe2a-d878-4dd6-80c3-6cd48b056dd&title=&width=198)

### 多头注意力
在transformer中，为了进一步增强模型的表征能力，会使用多个注意力头，也就是多头注意力，来对整个序列进行加权，具体的做法是分别使用h个线性层把Q、K、V从原始的维度$d_{model}$映射到$d_k$，就能得到h个Q, K, V，然后分别计算h次attention，最后把这些拼接起来，过一个线性层再映射回原来的维度$d_{model}$，如下图所示：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1683037323640-aa425494-682d-4127-a367-a1ecc87dd961.png#averageHue=%23faf9f9&clientId=ucb8551b1-5e38-4&from=paste&height=177&id=udd39dd4b&originHeight=354&originWidth=1651&originalType=binary&ratio=2&rotation=0&showTitle=false&size=84594&status=done&style=none&taskId=ud449b069-d10a-4758-b0d0-a23aa0ec9bd&title=&width=825.5)<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1683021827190-b7a9efc7-f27b-4748-b26c-18844199c7e5.png#averageHue=%23f1f1f0&clientId=ucb8551b1-5e38-4&from=paste&height=332&id=u8a8476ea&originHeight=664&originWidth=760&originalType=binary&ratio=2&rotation=0&showTitle=false&size=61201&status=done&style=none&taskId=u960fc399-b110-4089-bcea-f30efd47dde&title=&width=380)<br />具体到论文里面的细节，transformer使用了8个注意力头，$d_{model}$为512，$d_k$设置为$d_{model}/8=64$，虽然多头注意力多了一些权重矩阵，但是由于每个注意力头的维度只有64，并且是可以并行计算的，因此计算成本和不使用多头注意力是差不多的。而使用了多头注意力可以使得每个注意力头关注序列不同部分的信息，进而捕捉到不同的语义信息，比如有的注意力头可能关注语法，而有的关注句子结构等，从而有效地提高模型的性能。

### 注意力机制在模型中的应用
说完了Attention的计算方式，再回到transformer模型中，这里有三种不同的attention，主要的区别就是Q, K, V的不同：

1. encoder中的自注意力，在encoder的自注意力层中，所有的Q, K, V都是来自于输入的序列，所以称为自注意力，具体来说，要得到当前位置的自注意力输出，会使用当前位置的词元表征作为query，然后整个序列的词元作为key和value，然后进行多头注意力的计算，最终得到当前位置的输出。
2. decoder的自注意力，与encoder的自注意力相似，所有的Q, K, V都是来源于输入的序列，不过由于是翻译任务，在预测时的时候需要以自回归的方式一个个生成词元，因此在训练时需要屏蔽当前词元之后的词元，这里的具体做法就是在送入softmax之前，把当前词元的query和之后词元的key的缩放点积置为负无穷，这样，他们进入softmax计算得到的相似度就是0，通过这种方式来进行屏蔽。
3. decoder中的encoder-decoder注意力，注意这里的QKV就不是都来源于输入的序列了，而是Q来源于上一个decoder的输入序列，而K和V来源于encoder的输出序列，一般的seq2seq模型使用注意力机制也都是这么做的。

## 残差连接与LayerNorm
残差连接是来源于ResNet, 为了解决深度神经网络中的梯度消失和梯度爆炸问题, 我们这里的transformer由于网络深度也非常深, 因此也引入了残差连接<br />而LayerNorm与BatchNorm类似, 都是一种归一化的方法, 不过归一化的维度不同, BatchNorm在mini-batch中对每个特征维度进行归一化，使得得到值的均值和方差都接近于0和1, 计算的维度是特征这个维度, 假设有C个特征会得到C个特征的统计值, 而LayerNorm则是对每个样本的特征维度进行归一化，使得每个样本上的每个特征的均值和方差接近于0和1, 计算的维度是样本这个维度, 有N个样本的话就会得到N个样本的统计值<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1683044251788-9270e9f2-b90c-4b86-b94f-4050f1767ddb.png#averageHue=%237e7e77&clientId=ucb8551b1-5e38-4&from=paste&height=230&id=u30d76700&originHeight=170&originWidth=296&originalType=url&ratio=2&rotation=0&showTitle=false&size=13441&status=done&style=none&taskId=u5d211bc2-3472-4122-ac3d-1c5d2af708b&title=&width=400)<br />那么为什么在transformer里面要用LayerNorm而不用BatchNorm呢, 虽然 BatchNorm 可以在训练过程中缓解内部协变量移位的问题，但在处理序列数据时却存在一些问题。因为序列数据的长度通常是变化的，因此每个 mini-batch 的大小也是变化的。这意味着在训练过程中，每个 mini-batch 的统计信息可能会变化，从而导致 BatchNorm 的效果变差, 并且在测试时还需要维护均值和方差<br />相比之下，LayerNorm 是一种对每个样本中的每个特征维度进行归一化的技术，不受 mini-batch 大小的影响，因此更适合处理变长序列数据。另外，LayerNorm 不需要维护 mini-batch 统计信息，因此可以减少模型训练时的内存消耗，并且可以在测试时使用相同的归一化参数，从而避免了训练和测试时的不一致性。<br />在 Transformer 中，每个编码器和解码器层中的子层之间都使用了 LayerNorm，包括多头自注意力层和前馈网络层。这使得 Transformer 在处理序列数据时更加稳定和高效，并且可以在不同的任务中进行共享，提高了模型的泛化能力。因此，使用 LayerNorm 而不是 BatchNorm 是 Transformer 模型的一个重要设计选择。

## 基于位置的前馈神经网络
这个其实就是一个普通的两层的线性神经网络，分别作用于输入序列的每一个位置，参数共享，并且在过完第一层之后使用ReLU激活函数，输入维度为512，隐藏层维度为2048，输出维度再变为512，具体如下：<br />$\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$

## 输出层
在decoder输出时，使用了一个线性层+softmax，得到预测的每个位置上的词元的概率，并且这个线性层和encoder与decoder的两个嵌入层是共享参数矩阵的，同样的为了防止softmax可能导致的梯度消失，这里在计算时把嵌入层的权重乘以了$\sqrt{d_{model}
}$。

## 位置编码
由于自注意力对于所有的词元都是一视同仁的，不会考虑到位置上的信息，因此，为了能够捕捉到位置上的信息，transformer考虑在输入的embedding上面加上位置编码，这个位置编码有两种方式，一种是训练得到的，也就是你赋予每一个位置一个embedding，让模型自己学，另一种是使用固定的，也就是论文里面采用的方式，他这里使用了一个余弦函数：<br />$\begin{gathered}
P E_{(p o s,2i)} =sin(pos/10000^{2i/d_\mathrm{model}}) \\
PE_{(pos,2i+1)} =cos(pos/10000^{2i/d_\text{model}}) 
\end{gathered}$<br />其中pos是位置，i是维度。也就是说，位置编码的每个维度对应于一个正弦函数。<br />论文还试验了使用可学习的位置编码与固定的位置编码的效果，发现两个版本产生的结果几乎相同，而选择固定的正弦版本是因为它可以允许模型外推到比训练期间遇到的序列长度更长的序列长度。

# 为什么要使用自注意力？
文章的最后来探讨一下在序列建模任务中为什么要使用自注意力，这里和卷积以及RNN做了对比，分别从计算复杂度、可并行度，以及长序列建模能力来进行讨论。<br />![](https://cdn.nlark.com/yuque/0/2023/svg/764062/1683041720960-e8f41c4e-d26e-4128-8944-51e627b87cc5.svg#clientId=ucb8551b1-5e38-4&from=paste&id=u4f813c97&originHeight=419&originWidth=628&originalType=url&ratio=2&rotation=0&showTitle=false&status=done&style=none&taskId=uc7eea4d4-7ff0-4dc0-a47f-5ed489735e5&title=)<br />首先是计算复杂度，自注意力的计算复杂度可以说和卷积以及RNN的差不多，取决于输入的序列长度和embedding维度的大小，对于长序列计算attention会更为复杂。<br />然后是可并行度，这里的可并行度用的是以所需的最小顺序操作数来衡量，RNN建模需要满足先后关系，因此他是$O(n)$,而attention和cnn则是$O(1)$<br />最后是长序列建模能力，长序列的建模能力是用使用序列头尾之间相连的网络路径长度来计算的，attention可以直接通过计算头尾之间的相似度并且进行加权，因此他是$O(1)$, 而RNN的头尾则需要一步步传导，因此他是$O(n)$, 而CNN则是$O(log_k(n))$, 取决于卷积核的大小<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1683040501030-192c29d9-9c58-49c1-a95e-8ff01183053b.png#averageHue=%23f4f3f3&clientId=ucb8551b1-5e38-4&from=paste&height=196&id=u9d3a93fd&originHeight=391&originWidth=1632&originalType=binary&ratio=2&rotation=0&showTitle=false&size=93005&status=done&style=none&taskId=u040c5d01-b9ae-4ded-a2ea-abf29cbae2b&title=&width=816)<br />因此attention的好处在于他不仅计算复杂度可以接受, 并且由于可以并行, 工业界的模型都是在集群上并行计算的, 因此由于其并行能力强也可以看作他计算速度快了, 而且由于可以连接一个序列上的任意两个位置,对于长序列建模能力也很不错<br />还有一个就是他效果好, 而且由于可以输出注意力的分布, 使得他还具备一定的可解释性, 因此慢慢地在各个领域里面就都有应用了

# 参考

1. [Attention Is All You Need](http://arxiv.org/abs/1706.03762)
2. [《动手学深度学习》](https://zh-v2.d2l.ai/)
3. chatgpt
