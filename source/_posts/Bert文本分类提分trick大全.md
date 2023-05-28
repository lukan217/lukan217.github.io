---
title: Bert文本分类提分trick大全
mathjax: true
tags:
- 自然语言处理
- 深度学习
categories:
- 深度学习
---
作为一个0基础的小白，在过去的一年我也参加了若干场nlp相关的比赛，期间也学习了很多的trick，这些trick其实都有很多单独的文章介绍了，这里的话我主要做一个整合工作，结合自己比赛时的使用经验分享给大家，希望能够对大家的比赛或工作能够有帮助，这些trick并不局限于文本分类任务，对于一些更广泛的nlp场景，或者多模态和cv任务有的也都能够有不错的效果。

# 模型选型
如今huggingface上已经挂载了各式各样的bert，在做自己的任务的时候，选择哪个bert作为自己模型的backbone是非常重要的，因为这些bert它本身预训练的训练语料是多种多样的，可能和自己所需要做的任务天差地别，因此在做自己的文本分类任务的时候，首先一个就需要了解自己所做的任务是哪个领域的？是金融的、电商的还是UGC内容、新闻等等，然后还要了解自己的文本是中文、英文或者多语言，根据这些信息来选择合适的bert：

1. 如果是中文文本，可以优先考虑哈工大开源的两个模型：hfl/chinese-roberta-wwm-ext以及hfl/chinese-macbert-base，这两个在中文上的效果都非常不错
2. 如果是英文文本，可以直接考虑microsoft/deberta-v3-base，没啥好说的，kaggle最近的几个feedback比赛都是用的这个模型
3. 如果是一些垂直领域的任务，可以去搜一下huggingface上有没有相关的模型，如金融领域的finbert等，由于是在垂直领域的语料上进行训练，和自己的任务较为相近，因此一般来说效果都会比通用性的要好一点

另外，如果算力足够的话，可以考虑使用对应模型的large，一般来说也可以带来不错的收益。

# 预训练
bert是在大规模语料上进行预训练的，尽管这种预训练学习到了一些通用的能力，但是在自己的任务上表现还是差强人意，那么如何能够让bert更加适合自己的任务呢？答案就是领域内预训练，这里说的领域内预训练，就是在自己的训练集文本上执行预训练，训练结束后模型便更贴合自己的任务了，在训练的时候收敛速度加快，效果一般来说也能带来百分位点的提升，至于预训练的任务，那就分很多种了：

1. mlm预训练

也就是原生bert预训练两个任务（mlm与nsp）之一，一般也只做mlm，当然nsp任务在某些场景下可能也是有用的（个人还没试过），mlm任务做的就是完形填空，随机按照比例mask掉一些词，然后预测这个词可能是词表中某个词的概率，可以自己实现，不过如果单纯只做文本分类的话可以直接用huggingface提供的脚本：[https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py)<br />把这个脚本下载下来，然后准备好训练数据，写个训练脚本，就可以在本地得到预训练好的模型，直接引用即可。
```
export CUDA_VISIBLE_DEVICES=0
NUM_GPU=1
PORT_ID=$(expr $RANDOM + 1000)
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID  run_mlm.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --num_train_epochs 20 \
    --train_file ./data/train_pretrain.txt \
    --validation_file ./data/test_pretrain.txt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --output_dir ./save/pretrain_model \
    --line_by_line \
    --eval_steps 500  \
    --logging_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --save_total_limit 0 \
    "$@"
```

2. simcse预训练

这个预训练任务使用对比学习的方式，将一个句子过两次bert，由于dropout的存在得到的两个向量不完全一致，可以作为正样本，然后batch内其他向量作为负样本，使用infoCSE损失进行训练，也见到比赛里面一些大佬用过，具体训练方式可以参考：[https://github.com/princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE)

3. 多模态预训练

这个其实跟文本分类没啥关系了，主要是做多模态任务会将文本和图片/视频等其他模态的数据对齐，一般会用image text match，image text contrastive等损失，具体可以参考2022年微信大数据挑战赛的方案：[2022微信大数据挑战赛Top方案总结](https://zhuanlan.zhihu.com/p/567648297)。

# 对抗训练
对抗训练，也算是nlp和cv里面一种比较常用的提高模型鲁棒性的方法了，所谓对抗，就必须要有攻击和防守，通过对模型不断进行攻击，模型在防守的同时便能够提升自己的抗噪能力，也就是提高泛化能力。传统上对模型的攻击方式一般是通过制造对抗样本进行攻击，比如nlp可以通过对句子的某个词进行随机插入删除替换，cv可以对图片进行旋转裁剪缩放等操作，对样本进行扰动，以这种数据增强的方式来使得模型的鲁棒性得到提高。<br />而这里我们要讲的对抗训练的方式是对输入的embedding进行攻击，这样的好处便是不用自己再人为制定策略来对数据进行增强，而可以在训练过程中让攻击和防守自然而然的发生，具体的做法是这样的：

1. 攻击：在输入的embedding上进行梯度上升，使得loss变大
2. 防守：在参数上进行梯度下降，使得loss减少

这一过程会贯穿每一个训练step，因此便使得对抗的过程自然而然的发生，模型也会训练的越来越强，而具体的做法就有很多种了，如FGM,PGD, 还有最近在kaggle上比较流行的AWP，这里只介绍FGM的实现，因为比较简单，实战里用FGM也就足够了，一般来说也能够带来一到两个百分位点的提升：
```
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
# 初始化
fgm = FGM(model)
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward() # 反向传播，得到正常的grad
    # 对抗训练
    fgm.attack() # 在embedding上添加对抗扰动
    loss_adv = model(batch_input, batch_label)
    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    fgm.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()

```

# R-DROP
bert里面是加了dropout的，有隐藏层的dropout，也有attention的dropout，这些都会在微调时为模型带来一定的扰动，由于存在训练和预测时的不一致性，往往都会影响模型效果，因此就有论文想要通过给模型添加扰动，但是又使得这个扰动不那么大，这个就是R-Drop（Regularized Dropout for Neural Networks），论文的思想也很简单，就是把输入过两次神经网络，由于dropout的存在，最终输出的logits是不一样的，我们想要让他输出的logits更相似一点，就在损失函数里面再加一部分衡量两次输出logits相似度的kl散度：<br />$\begin{aligned}
\mathcal{L}^i=\mathcal{L}^i_{NLL}+\alpha\cdot\mathcal{L}^i_{KL}& =-\log\mathcal{P}_1^w(y_i|x_i)-\log\mathcal{P}_2^w(y_i|x_i)  \\
&+\frac{\alpha}{2}[\mathcal{D}_{KL}(\mathcal{P}_1^w(y_i|x_i)||\mathcal{P}_2^w(y_i|x_i))+\mathcal{D}_{KL}(\mathcal{P}_2^w(y_i|x_i)||\mathcal{P}_1^w(y_i|x_i))],
\end{aligned}$<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1685213138940-47e5bda5-44b6-40ae-a9a4-229a8c890392.png#averageHue=%23eae7e5&clientId=u006289c1-9ad5-4&from=paste&height=328&id=u861031ed&originHeight=656&originWidth=1485&originalType=binary&ratio=2&rotation=0&showTitle=false&size=147077&status=done&style=none&taskId=ub05d2f68-6d62-49b0-9bab-739fad667a5&title=&width=742.5)<br />原文也提供了代码实现，不过由于要过两次bert，因此训练时长和显存占用也要变多。
```
model = TaskModel()

def compute_kl_loss(self, p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

# keep dropout and forward twice
logits = model(x)

logits2 = model(x)

# cross entropy loss for classifier
ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))

kl_loss = compute_kl_loss(logits, logits2)

# carefully choose hyper-parameters
loss = ce_loss + α * kl_loss
```

# Multi Sample Dropout
R-Drop的实现虽然能够缓解DropOut带来的问题，但是要去改损失函数，并且可能还要调个合适的超参，Multi Sample Dropout就带来了一种相对优雅的实现方式，就是在输出层先多次dropout，再过线性层，然后把输出的logits和loss取平均，相比于R-Drop就不用去改损失函数加上KL散度了，而是直接暴力取平均<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1685216210275-18b35f2d-76cd-4425-8499-0e38d574c92c.png#averageHue=%23f8f6f6&clientId=u006289c1-9ad5-4&from=paste&height=387&id=u9f5ff9b4&originHeight=774&originWidth=1368&originalType=binary&ratio=2&rotation=0&showTitle=false&size=120083&status=done&style=none&taskId=ud92ade1a-1997-419b-8345-75c26544f55&title=&width=684)<br />这里我提供一种只把logits取平均的做法（这里没有把loss取平均，嫌麻烦），可以直接替换成输出层，即插即用
```
class MultiSampleClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, dropout=0.2, num=5):
        super(MultiSampleClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_labels)
        self.num = num
        self.dropout_ops = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num)
        ])

    def forward(self, x):
        logits = None
        for i, dropout_op in enumerate(self.dropout_ops):
            if i == 0:
                out = dropout_op(x)
                logits = self.linear(out)
            else:
                temp_out = dropout_op(x)
                temp_logits = self.linear(temp_out)
                logits += temp_logits
        # 相加还是取平均
        # if self.args.ms_average:
        logits = logits / self.num
        return logits
```

# 去掉DropOut
前面提到的两种方法，无论是R-Drop还是Multi Sample Dropout，都是为了解决微调时DropOut训练和测试存在不一致的问题，这个问题我之前在一篇文章里面讲过，在回归问题会有比较大的偏差，最近我在分类问题中也观察到了这样的情况，怪不得这两种方法会那么流行。<br />回归问题的本质，先思考一下DropOut为什么在bert微调时有问题？我个人的猜测是DropOut虽然是一种缓解过拟合的方法，但是可能并不适合bert微调的场景，首先是因为网络层数比较深了，bert在每层上面都加了dropout，每一层都带有扰动的话，经过逐层放大，这样就会导致最终的输出非常不稳定，其次，bert因为是预训练过的，一般来说只需要用较小的学习率微调几轮很快便收敛了，而dropout可能更适合那种需要训练较多轮才能发挥它的作用，因此DropOut在bert文本分类场景下可能并不好用。<br />那么既然dropout会存在问题，为什么还需要用r-drop，Multi Sample Dropout这种花里胡哨的方法来缓解他呢？直接在训练时去掉不行吗？经过我最近一次实验的尝试，去掉dropout，确实也能够在分类问题上得到提升，不过这个结论有待进一步验证，目前去掉dropout给我带来的就是网络收敛更快了，而且效果上也好上一两个点，比r-drop和Multi Sample Dropout效果要更好，而具体实现方式就非常简单了，只需要两行代码，所以非常建议大家也去试下，看看这个结论是否可靠：
```
self.config = AutoConfig.from_pretrained(args.bert_dir)
self.config.hidden_dropout_prob = 0.
self.config.attention_probs_dropout_prob = 0.
self.bert = AutoModel.from_pretrained(args.bert_dir,config=self.config)
```
中间两行就是关键代码，直接把bert的dropout在训练时都关闭了

# 学习率相关
bert对学习率是非常敏感的，稍微大一点模型就训不起来了，一般学习率区间设置在1e-5~5e-5就可以了，这是全局的学习率设置，对所有层都一视同仁，然而由于bert是一个深层的在大规模语料上训练过的模型，因此，bert的底层其实已经学到了一些通用的能力，这部分学习率就不需要设置的那么高，而在bert顶层，这部分是task specific的，就需要设置比较大的学习率，因此便可以使用不同层不同学习率的方式，比如输出层用5e-5，而bert使用2e-5，也有一些比较花的设置，学习率逐层递减，不同层设置不同学习率，也可以带来一点点的提升。<br />除了学习率的设置，学习率的调度也可以带来一点提升，个人比较常用的调度器是CosineAnnealingWarmRestarts，因为学习率是周期变化的，就不需要太考虑一些超参数，比如要训练多少步这种，用起来比较方便，并且在大部分场景下都可以得到不错的效果：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/764062/1685264235675-5b5acfb9-7828-49b5-9d16-a39eba66afe1.png#averageHue=%23f9f9f9&clientId=u006289c1-9ad5-4&from=paste&id=u6befcb62&originHeight=360&originWidth=720&originalType=url&ratio=2&rotation=0&showTitle=false&size=34917&status=done&style=none&taskId=ub71960c9-1a20-4056-9180-ef398d4120d&title=)


# 长文本处理
实践中经常要对一些篇章级的文本进行处理，但是大部分bert的输入最多只能支持512个字符，而篇章级的文本往往都是上千个字符，因此就需要一些手段来处理这种长文本输入，处理好了往往能够带来几个点的提升，具体的处理方法我可以分为三种吧：

1. 从文本中提取最关键的信息作为输入，既然bert只能限制512个字符，那么就从文本中挑选出512个来就好了，这个具体的处理方式就是task specific的了，一般来说，对于一个文章分类，你把他的标题、第一段内容、最后一段内容作为输入便可以得到比较好的效果，因为标题和首尾一般来说都会含有比较关键的信息，这个也是个人实践中提点最快的方法
2. 选择可以处理长文本的模型，比如LongFormer，BigBird，deberta等，这些模型都可以突破512个字符的限制，不过说实话带来的提升较为有限，甚至是副作用，因为主流比较强的模型还是仅支持输入512个字符的模型
3. 拆分句子多次过bert，思想很简单，既然只能处理512个字符，那么我就把句子拆开，0-512送进bert得到一个隐藏层向量，512-1024再送进bert得到另一个向量，依此类推，最后再pooling一下，不过这种方法的缺点是会损失位置编码的信息

# Pooling方式
前面说的长文本处理有一部分可以说是选择什么样的输入进入到bert编码器中，但是经过bert编码之后得到的输出是形状[batch_size,seq_len,hidden_size]，需要把seq_len这个维度去掉再接输出层，这时候面临的问题是选择一个什么样的输出接输出层，也就是我们说的Pooling池化，传统bert是直接把第一个位置，也就是[CLS]对应的向量作为输出，这样做其实很不好，尤其是在文本分类上，只用第一个位置字符的输出损失了相当多的信息，实践中用这种方式也会比其他方法低几个点，所以一般最好使用MeanPooling获取全局的语义信息，或者使用MaxPooling获取关键信息，最近我在一个任务上尝试了Attention Pooling，也就是为每个位置算一个Attention权重，然后对每个位置进行加权得到最后的输出，这种方式在我的任务上效果非常不错，收敛也很快，实现如下：
```
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )
    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings
```

# EMA
Exponential Moving Average（指数移动平均），主要思想是对模型进行“自融合”，自己和自己融合的意思，融合方式是通过对模型不同训练step时的模型参数取加权平均，加权的权重如何设置呢？就是用指数移动平均的方式，对近期的参数取较高的权重，而对以前的权重取较低的权重，经过加权平均后，模型的参数更容易落在全局最优点，泛化性更高，具体实现如下，也是即插即用的，不过对于训练step比较少的场景，不一定能够带来提升，并且，最好不要在训练开始的时候就用ema，因为这样训练前期的收敛很慢，可以等快要收敛的时候才开始ema
```
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 初始化
ema = EMA(model, 0.999)
ema.register()

# 训练过程中，更新完参数后，同步update shadow weights
def train():
    optimizer.step()
    ema.update()

# eval前，apply shadow weights；eval之后，恢复原来模型的参数
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()

```

# 伪标签
伪标签在很多比赛的top方案上也出现过，好像最后也能提一两个点，需要模型能够具有较高的预测精度了再使用效果才比较好，主要思想也比较简单：

1. 在训练集上训练模型，并预测测试集的标签
2. 取测试集中预测置信度较高的样本（如预测为1的概率大于0.95），加入到训练集中
3. 使用新的训练集重新训练一个模型，并预测测试集的标签
4. 重复执行2和3步骤若干次（一至两次即可）

自己之前用过一次没效果，后面就没再用了，没啥成功的经验，下面贴一段chatgpt的实现：
```
# 准备已标记和未标记的数据集
labeled_dataset = ...  # 已标记数据集
unlabeled_dataset = ...  # 未标记数据集

# 创建数据加载器
labeled_dataloader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True)

# 初始化模型
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始训练，使用已标记数据
model.train()
for epoch in range(5):  # 初始训练若干轮
    for inputs, labels in labeled_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 使用伪标签进行训练
model.train()
for epoch in range(5):  # 使用伪标签训练若干轮
    pseudo_labels = []
    unlabeled_data = []
    for inputs, _ in unlabeled_dataloader:
        outputs = model(inputs)
        pseudo_labels.extend(torch.argmax(outputs, dim=1).tolist())
        unlabeled_data.append(inputs)

    # 将伪标签与未标记数据合并，形成新的标记数据集
    pseudo_labeled_dataset = torch.utils.data.TensorDataset(torch.cat(unlabeled_data, dim=0), torch.tensor(pseudo_labels))
    pseudo_labeled_dataloader = DataLoader(pseudo_labeled_dataset, batch_size=32, shuffle=True)

    for (inputs, labels), (pseudo_inputs, pseudo_labels) in zip(labeled_dataloader, pseudo_labeled_dataloader):
        optimizer.zero_grad()
        labeled_outputs = model(inputs)
        pseudo_labeled_outputs = model(pseudo_inputs)
        labeled_loss = criterion(labeled_outputs, labels)
        pseudo_labeled_loss = criterion(pseudo_labeled_outputs, pseudo_labels)
        loss = labeled_loss + pseudo_labeled_loss
        loss.backward()
        optimizer.step()
```

# 模型融合
模型融合，算是比赛上分的最后手段了，具体可以有单模型融合和多模型融合：

1. 单模型融合：也就是五折交叉融合，同一个模型，在五折的数据上训练，然后将预测结果进行融合，由于每一折的数据都不太一样，所以给预测结果带来了一定的差异性，融合可以上一点点分
2. 多模型融合：在不同的模型上进行训练，然后将预测结果再进行融合，这里需要注重模型的差异性要足够大，才能够带来足够的提升，具体做法可以是在模型选型上（roberta，roberta large，macbert等），pooling方式上（max/mean/attention pooling），文本输入上（不同阶段方式）等进行排列组合，最后top分数之间的差异很可能就是融合的模型数量的差异，因为基本的东西大家都做的差不多，最后就是拼算力，看谁能搞出足够多的模型来融合。

具体到融合的手段，可以按照概率融合也可以按照投票融合，两者应该是没有显著的差异的，概率融合的话更方便为比较好的模型赋予更高的权重。


# 参考

1. [【炼丹技巧】功守道：NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)
2. [R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)
3. [Multi-Sample Dropout for Accelerated Training and Better Generalization](https://arxiv.org/abs/1905.09788)
4. [【炼丹技巧】指数移动平均（EMA）的原理及PyTorch实现](https://zhuanlan.zhihu.com/p/68748778)
