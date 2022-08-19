---
title: Transformers的性能优化方法
mathjax: true
tags:
- nlp
categories: 
- 机器学习
---

# 前言

自BERT出现以来，nlp领域已经进入了大模型的时代，大模型虽然效果好，但是毕竟不是人人都有着丰富的GPU资源，在训练时往往就捉襟见肘，出现显存out of memory的问题，或者训练时间非常非常的久，因此，这篇文章主要解决的问题就是如何在GPU资源受限的情况下训练transformers库上面的大模型。<br />这篇文章源自[@Vadim Irtlach](https://www.kaggle.com/vad13irt)大佬在kaggle的[开源notebook](https://www.kaggle.com/code/vad13irt/optimization-approaches-for-transformers/notebook)，感谢原作者的分享，本nlp小白觉得受益良多，因此搬运到知乎分享给大家，已取得作者授权，大部分内容是照搬翻译过来的，小部分内容结合自己的理解进行了补充和修改，不对的地方请大家批评指正，正文开始！

---

尽管Huggingface开源的Transformers在自然语言处理（NLP）任务中取得了惊人的成功，但由于里面的模型参数数量庞大，即使是使用GPU进行训练或者部署，也仍具有非常大的挑战性，因为用如此大的模型进行训练或推理，会很容易发生显存不足（OOM）以及训练时间过长的问题。（这里想吐槽一句的是，kaggle上面的nlp比赛现在动不动就用五折debert-large-v3，没几块V100根本玩不起这种比赛，所以这篇文章对我这种只能用colab的p100来跑实验的穷学生来说真的是福音啊！）<br />然而，有很多方法可以避免显存不足以及训练时间过长的方法，这篇文章的主要贡献就是介绍了这些方法的原理以及如何实现，具体包括以下几种方法：

1. 梯度累积（Gradient Accumulation）
2. 冻结（Freezing）
3. 自动混合精度（Automatic Mixed Precision）
4. 8位优化器（8-bit Optimizers）
5. 梯度检查点（Gradient Checkpointing）
6. 快速分词器（Fast Tokenizers）
7. 动态填充（Dynamic Padding）
8. 均匀动态填充（Uniform Dynamic Padding）

其中1-5是神经网络通用的方法，可以用在任何网络的性能优化上，6-8是针对nlp领域的性能优化方法。

# 梯度累积

梯度累积背后的想法非常简单，就是为了模拟更大的批量（batch）。有时，为了更好地收敛或提高性能，需要使用大批量进行训练，但是，这通常需要更大的显存。这个问题的一种可能的解决方案是使用较小的批量，但是，一方面，小批量训练会增加训练和推理时间，另一方面，梯度下降算法对批量大小的选择非常敏感，小批量可能会导致不稳定的收敛和性能降低。所以，我们可以先执行几次前向传播和反向传播，使得梯度进行累积，当我们有足够的计算梯度时，再对参数进行优化，从而利用小显存，模拟大批量的效果，并且训练时间也不会大幅增加。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1660620493927-e9036332-e9f7-4f8c-b197-cef9a4957a11.png#clientId=u95242c66-2c25-4&crop=0&crop=0&crop=1&crop=1&from=paste&id=u13064602&margin=%5Bobject%20Object%5D&name=image.png&originHeight=998&originWidth=1400&originalType=url&ratio=1&rotation=0&showTitle=false&size=135231&status=done&style=none&taskId=u76579be5-a79c-47e5-906d-00a8b32b705&title=)

## 代码实现

```python
steps = len(loader)

# perform validation loop each `validation_steps` training steps!
validation_steps = int(validation_steps * gradient_accumulation_steps)

for step, batch in enumerate(loader, 1):

    # prepare inputs and targets for the model and loss function respectively.

    # forward pass
    outputs = model(inputs)

    # computing loss
    loss = loss_fn(outputs, targets)

    # accumulating gradients over steps
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps

    # backward pass
    loss.backward()

        # perform optimization step after certain number of accumulating steps and at the end of epoch
    if step % gradient_accumulation_steps == 0 or step == steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        model.zero_grad()

            # perform validation loop
    if step % validation_steps == 0:
        validation_loop()
```

# 冻结

冻结是一种非常有效的方法，通过取消计算模型某些层中的梯度计算（如embedding层，bert的前几层），可以大大加快训练速度并且降低了显存占用，而且几乎不会损失模型的性能。<br />深度学习中的一个众所周知的事实是，网络的底层学习输入数据的通用特征，而网络顶层学习目标任务特定的高级特征，所以在对预训练模型进行微调时，一般网络底层的参数都不怎么需要变，这些都是通用的知识，需要学习的是顶层的那些参数，当使用某种优化算法（如SGD、AdamW或RMSprop）执行优化步骤时，网络的底层的梯度就都很小，因此参数几乎保持不变，这也被称为梯度消失，因此，与其花费大量的时间和算力来计算底层这些“无用”梯度，并对此类梯度很小的参数进行优化，不如直接冻结它们，直接不计算梯度也不进行优化。<br />PyTorch为关闭梯度计算提供了一个舒适的API，可以通过`torch.Tensor`的属性`requires_ grad`设置。

## 代码实现

```python
def freeze(module):
    """
    Freezes module's parameters.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False

def get_freezed_parameters(module):
    """
    Returns names of freezed parameters of the given module.
    """

    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)

    return freezed_parameters
```

```python
import torch
from transformers import AutoConfig, AutoModel


# initializing model
model_path = "microsoft/deberta-v3-base"
config = AutoConfig.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, config=config)


# freezing embeddings and first 2 layers of encoder
freeze(model.embeddings)
freeze(model.encoder.layer[:2])

freezed_parameters = get_freezed_parameters(model)
print(f"Freezed parameters: {freezed_parameters}")

# selecting parameters, which requires gradients and initializing optimizer
model_parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(params=model_parameters, lr=2e-5, weight_decay=0.0)
```

# 自动混合精度

自动混合精度（AMP）是另一种在不损失最终质量的情况下减少显存消耗和训练时间的方法，该方法由NVIDIA和百度研究人员在2017年的["Mixed Precision Training"](https://arxiv.org/abs/1710.03740)论文中提出。该方法背后的关键思想是使用较低的精度将模型的梯度和参数保留在内存中，即不使用全精度（float32），而是使用半精度（例如float16）将张量保存在内存中。然而，当以较低精度计算梯度时，某些值可能太小，以至于被视为零，这种现象被称为“溢出”。为了防止“溢出”，原始论文的作者提出了一种梯度缩放方法。<br />PyTorch从1.6的版本开始提供了一个包：`torch.cuda.amp`，具有使用自动混合精度所需的功能（从降低精度到梯度缩放），自动混合精度作为上下文管理器实现，因此可以随时随地的插入到训练和推理脚本中。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1660620741318-d97b8b39-dbb2-4147-b649-3a0276578385.png#clientId=u95242c66-2c25-4&crop=0&crop=0&crop=1&crop=1&from=paste&id=u98636977&margin=%5Bobject%20Object%5D&name=image.png&originHeight=711&originWidth=1512&originalType=url&ratio=1&rotation=0&showTitle=false&size=244947&status=done&style=none&taskId=ub87efba5-33dd-4317-8b04-f6461940227&title=)

## 代码实现

```python
from torch.cuda.amp import autocast, GradScaler


scaler = GradScaler()

for step, batch in enumerate(loader, 1):

    # prepare inputs and targets for the model and loss function respectively.

    # forward pass with `autocast` context manager
    with autocast(enabled=True):
        outputs = model(inputs)

    # computing loss
    loss = loss_fn(outputs, targets)

    # scale gradint and perform backward pass
    scaler.scale(loss).backward()

    # before gradient clipping the optimizer parameters must be unscaled.
    scaler.unscale_(optimizer)

    # perform optimization step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    scaler.step(optimizer)
    scaler.update()
```

# 8-bit Optimizers

8-bit Optimizers的思想类似于自动混合精度（模型的参数和梯度使用较低的精度保存），但8-bit Optimizers还让优化器的状态使用低精度保存。作者（Meta Research）在最初的论文["8-bit Optimizers via Block-wise Quantization"](https://arxiv.org/abs/2110.02861)中详细介绍了8-bit Optimizers，表明8-bit Optimizers显著降低了显存占用，略微加快了训练速度。此外，作者研究了不同超参数设置的影响，表明8-bit Optimizers对不同的学习率、beta和权重衰减参数的效果是稳定的，不会降低性能或影响收敛性。因此，作者为8位优化器提供了一个高级库，叫做[bitsandbytes](https://github.com/facebookresearch/bitsandbytes)。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1660804734860-c3a9d710-3d76-497f-8a44-3637278eb344.png#clientId=ua760d25a-e986-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=490&id=u5caea8f7&margin=%5Bobject%20Object%5D&name=image.png&originHeight=979&originWidth=1365&originalType=binary&ratio=1&rotation=0&showTitle=false&size=347195&status=done&style=none&taskId=uc97d0d51-2866-4af1-a7b1-7683d0384c8&title=&width=682.5)

## 代码实现

```python
!pip install -q bitsandbytes-cuda110
```

```python
def set_embedding_parameters_bits(embeddings_path, optim_bits=32):
    """
    https://github.com/huggingface/transformers/issues/14819#issuecomment-1003427930
    """

    embedding_types = ("word", "position", "token_type")
    for embedding_type in embedding_types:
        attr_name = f"{embedding_type}_embeddings"

        if hasattr(embeddings_path, attr_name): 
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                getattr(embeddings_path, attr_name), 'weight', {'optim_bits': optim_bits}
            )

import bitsandbytes as bnb


# selecting parameters, which requires gradients
model_parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())

# initializing optimizer 
bnb_optimizer = bnb.optim.AdamW(params=model_parameters, lr=2e-5, weight_decay=0.0, optim_bits=8)
# bnb_optimizer = bnb.optim.AdamW8bit(params=model_parameters, lr=2e-5, weight_decay=0.0) # equivalent to the above line

# setting embeddings parameters
set_embedding_parameters_bits(embeddings_path=model.embeddings)

print(f"8-bit Optimizer:\n\n{bnb_optimizer}")
```

# 梯度检查点

有时候，即使用了上面的几种方法，显存可能还是不够，尤其是在模型足够大的情况下。那么梯度检查点（Gradient Checkpointing）就是压箱底的招数了，这个方法第一次在 ["Training Deep Nets With Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) ，作者表明梯度检查点可以显著降低显存利用率，从$O(n)$降低到$O(\sqrt n)$，其中n是模型的层数。这种方法允许在单个GPU上训练大型模型，或者提供更多内存以增加批量大小，从而更好更快地收敛。梯度检查点背后的思想是在小数据块中计算梯度，同时在正向和反向传播过程中从内存中移除不必要的梯度，从而降低内存利用率，但是这种方法需要更多的计算步骤来再现整个反向传播图，其实就是一种用时间来换空间的方法。<br />![0_nMSeZxl6ppnrivgv_.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1660805746716-029ea553-21a5-4eeb-8488-988eba2d47df.png#clientId=ua760d25a-e986-4&crop=0&crop=0&crop=1&crop=1&from=drop&id=u21665c21&margin=%5Bobject%20Object%5D&name=0_nMSeZxl6ppnrivgv_.png&originHeight=730&originWidth=1400&originalType=binary&ratio=1&rotation=0&showTitle=false&size=107465&status=done&style=none&taskId=ua39d3d57-9858-41c6-8671-e8d98f729b4&title=)

![0_s7U1QDfSXuVd1LrF_.gif](https://cdn.nlark.com/yuque/0/2022/gif/764062/1660805738813-5c578b58-806f-406b-a280-78d35db405fc.gif#clientId=ua760d25a-e986-4&crop=0&crop=0&crop=1&crop=1&from=drop&id=u532bec1a&margin=%5Bobject%20Object%5D&name=0_s7U1QDfSXuVd1LrF_.gif&originHeight=121&originWidth=541&originalType=binary&ratio=1&rotation=0&showTitle=true&size=309943&status=done&style=none&taskId=u967c6883-a688-4a8a-8dd9-1a15a4dc3a7&title=%E6%BC%94%E7%A4%BA%E6%A2%AF%E5%BA%A6%E6%A3%80%E6%9F%A5%E7%82%B9%E5%A6%82%E4%BD%95%E5%9C%A8%E6%AD%A3%E5%90%91%E5%92%8C%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E8%BF%87%E7%A8%8B%E4%B8%AD%E5%B7%A5%E4%BD%9C "演示梯度检查点如何在正向和反向传播过程中工作")<br />PyTorch框架里也有梯度检查点的实现，通过这两个函数：`torch.utils.checkpoint.checkpoint`和`torch.utils.checkpoint.checkpoint_sequential`<br />这边引用一段torch官网对梯度检查点的介绍：

> 梯度检查点通过用计算换取内存来工作。检查点部分不是存储整个计算图的所有中间激活以进行反向计算，而是不保存中间激活，而是在反向过程中重新计算它们。它可以应用于模型的任何部分。
> 具体而言，在前向传播中，该函数将以torch.no_grad()的方式运行，即不存储中间激活。然而，前向传播保存了输入元组和函数参数。在反向传播时，检索保存的输入和函数，然后再次对函数进行前向传播，现在跟踪中间激活，然后使用这些激活值计算梯度。

此外，HuggingFace Transformers也支持梯度检查点。梯度检查点可以通过PreTrainedModel实例的gradient_checkpointing_enable方法执行，一行代码直接搞定！

## 代码实现

```python
from transformers import AutoConfig, AutoModel
# https://github.com/huggingface/transformers/issues/9919
from torch.utils.checkpoint import checkpoint

# initializing model
model_path = "microsoft/deberta-v3-base"
config = AutoConfig.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, config=config)

# gradient checkpointing
model.gradient_checkpointing_enable()
print(f"Gradient Checkpointing: {model.is_gradient_checkpointing}")
```

# 快速分词器

HuggingFace Transformers提供两种类型的分词器：基本分词器和快速分词器。它们之间的主要区别在于，快速分词器是在Rust上编写的，因为Python在循环中非常慢，但在分词的时候又要用到循环。快速分词器是一种非常简单的方法，允许我们在分词的时候获得额外的加速。要使用快速分词器也很简单，只要把[transformers.AutoTokenizer](https://huggingface.co/docs/transformers/v4.19.3/en/model_doc/auto#transformers.AutoTokenizer) 里面的[from_pretrained](https://huggingface.co/docs/transformers/v4.19.3/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)方法的`use_fast`的值修改为True就可以了。<br />![分词器是如何工作的](https://cdn.nlark.com/yuque/0/2022/svg/764062/1660807996929-7d3dd145-9e93-407e-a30c-25b13431b15b.svg#clientId=ua760d25a-e986-4&crop=0&crop=0&crop=1&crop=1&from=paste&id=u7912c397&margin=%5Bobject%20Object%5D&originHeight=1276&originWidth=1776&originalType=url&ratio=1&rotation=0&showTitle=true&status=done&style=none&taskId=uc1ee8d1c-28f5-4757-9c5a-8f6976a03ed&title=%E5%88%86%E8%AF%8D%E5%99%A8%E6%98%AF%E5%A6%82%E4%BD%95%E5%B7%A5%E4%BD%9C%E7%9A%84 "分词器是如何工作的")

## 代码实现

```python
from transformers import AutoTokenizer

# initializing Base version of Tokenizer
model_path = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
print(f"Base version Tokenizer:\n\n{tokenizer}", end="\n"*3)

# initializing Fast version of Tokenizer
fast_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
print(f"Fast version Tokenizer:\n\n{fast_tokenizer}")
```

# 动态填充

通常来说，模型是用批量数据输入训练的，批中的每个输入必须具有固定大小，即一批量的数据必须是矩阵的表示，所有批量数据的尺寸都一样。固定尺寸通常是根据数据集中的长度分布、特征数量和其他因素来选择的。在NLP任务中，输入大小称为文本长度，或者最大长度（max length）。然而，不同的文本具有不同的长度，为了处理这种情况，研究人员提出了填充标记和截断。当最大长度小于输入文本的长度时，会使用截断，因此会删除一些标记。当输入文本的长度小于最大长度时，会将填充标记，比如[PAD]，添加到输入文本的末尾，值得注意的是，填充标记不应包含在某些任务的损失计算中（例如掩蔽语言建模或命名实体识别）<br />![fixed_padding_length.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1660809140323-9d6e9dc1-296f-4a0b-b311-e8b9d817a936.png#clientId=ua760d25a-e986-4&crop=0&crop=0&crop=1&crop=1&from=drop&id=udb05c024&margin=%5Bobject%20Object%5D&name=fixed_padding_length.png&originHeight=580&originWidth=1581&originalType=binary&ratio=1&rotation=0&showTitle=true&size=114729&status=done&style=none&taskId=uf50757a8-36d6-4b19-a2d9-87f7b181e13&title=%E5%9B%BA%E5%AE%9A%E9%95%BF%E5%BA%A6%E5%A1%AB%E5%85%85 "固定长度填充")<br />然而，填充标记有明显的缺点。比如在输入文本相对于选定的最大长度非常短的情况下，效率就很低，需要更多的额外内存，比如我有一条文本长度512，然后其他文本长度都在10左右，那么如果将max seq设置为512，就会导致很多无效计算。为了防止额外的计算操作，研究人员提出了一种非常有效的方法，就是将批量的输入填充到这一批量的最大输入长度，如下图所示，这种方法可以将训练速度提高35%甚至50%，当然这种方法加速的效果取决于批量的大小以及文本长度的分布，批量越小，加速效果越明显，文本长度分布越不均，加速效果也越好。<br />![dynamic_padding.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1660809160291-199358bc-6c29-4aca-b82e-1f5ca4b1d5d8.png#clientId=ua760d25a-e986-4&crop=0&crop=0&crop=1&crop=1&from=drop&id=u8b70e8fd&margin=%5Bobject%20Object%5D&name=dynamic_padding.png&originHeight=577&originWidth=1582&originalType=binary&ratio=1&rotation=0&showTitle=true&size=107538&status=done&style=none&taskId=u597d4c1c-1d38-40d6-844d-45ec13aa727&title=%E5%8A%A8%E6%80%81%E5%A1%AB%E5%85%85 "动态填充")

# 均匀动态填充

还有一种基于动态填充的方法，叫做均匀动态填充。其思想是在分batch时，先按文本的长度对文本进行排序，这样同一个batch里面的文本长度就都差不多。这种方法非常有效，在训练或推理期间的计算量都比动态填充要来的少。但是，不建议在训练期间使用均匀动态填充，因为训练时数据最好是要shuffer的，但是推理时如果一次性要推理很多文本的话可以考虑这么做

![uniform_length_batching.png](https://cdn.nlark.com/yuque/0/2022/png/764062/1660814142931-1222a4b1-82ad-4816-a313-8ef565a5f1bb.png#clientId=ua760d25a-e986-4&crop=0&crop=0&crop=1&crop=1&from=drop&id=u04146ac5&margin=%5Bobject%20Object%5D&name=uniform_length_batching.png&originHeight=585&originWidth=1592&originalType=binary&ratio=1&rotation=0&showTitle=true&size=101412&status=done&style=none&taskId=u939a8674-40c3-4a4d-9643-629ab22e05f&title=%E5%9D%87%E5%8C%80%E5%8A%A8%E6%80%81%E5%A1%AB%E5%85%85 "均匀动态填充")

# 总结

即使在现代GPU上，优化内存和时间也是开发模型的必要步骤，因此，本文介绍了加速训练和减少transformers等大型模型内存消耗的最强大、最流行的方法。

# 参考

1. [Performance and Scalability: How To Fit a Bigger Model and Train It Faster](https://huggingface.co/docs/transformers/performance)
2. [Speeding up Transformer w/ Optimization Strategies](https://www.kaggle.com/code/rhtsingh/speeding-up-transformer-w-optimization-strategies)
3. [Things you can try to speed up training speed and preventing memory shortage if you are using transformers.](https://www.kaggle.com/competitions/AI4Code/discussion/327777)
4. [8-bit Adam and other memory optimizations](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/303131)
5. [Fitting larger networks into memory.](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)
