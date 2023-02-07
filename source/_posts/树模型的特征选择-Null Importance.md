---
title: 树模型的特征选择-Boruta
mathjax: true
tags:
- 机器学习
- 特征选择
categories:
- 机器学习
---

Null Importance，和很久之前介绍过的[Boruta](https://zhuanlan.zhihu.com/p/479950999)一样，是一种专门针对树模型的特征选择方法，用来解决树模型中特征重要性并不能真实反映特征与标签的真实关系的问题。最近在一些特征选择的问题上用到了它，效果还不错，并且速度和内存占用上比Bortuta要优秀很多。
<a name="Ca4V7"></a>
# 算法思想
Null Importance的核心思想是输出两种特征重要性，一种是实际的特征重要性，另一种是打乱标签训练得到的null importance，然后使用后者对前者做一个修正，以得到真实的特征重要性。<br />具体步骤：

1. 计算实际的importance，以正常的方式训练一个树模型，输出特征重要度，这个特征重要度便反映了模型训练过程中起作用的特征排序。
2. 计算null importance，首先将标签打乱，然后以步骤一同样的方式训练一个模型，使用模型输出特征重要性，这里输出的特征重要性就被称为null importance，这个null importance就反映了在不考虑标签的情况下，模型在训练过程中是如何理解特征的。然而，由于特征被打乱过，特征与目标之间已经不存在实际意义上的关联，所以结果是不稳定的，就需要多次实验来得到一个null importance的分布才会比较准确
3. 计算特征分数，对于好的特征，实际的importance应该会很高，而null importance会很低，而对于不好的特征，它的null importance会和实际的importance差不多甚至大于实际的importace，因此我们便可以借助步骤12输出的实际importance和null importance的相对关系来构建一个评价体系，也就是每个特征的分数，来评价一个特征好不好，具体做法就有很多了，比如实际importance与null importance的差值、比率等，作者这里使用了log(实际importance/null importance的75分位数)
4. 输出特征选择结果，根据计算的特征分数，便可以卡一个阈值，来选择需要输出的特征数量，也可以自己观察下实际的importance和null importance的差距，来决定特征的去留。

从上面的过程中我们可以发现，null importance这种方法实际上是对实际的特征重要性做了一个修正，之前提到特征重要性不能用来做特征选择的一个问题便是，高基数的类别特征或者连续性特征特征重要性天然会比别的特征大，因为特征重要度是根据决策树分裂前后节点的不纯度的减少量（基尼系数或者MSE）来算的，对于数值特征或者类别多的的类别特征，不纯度较少相对来说会比较多，而且像在lgb和xgb这种模型中，如果指定了category feature，这些特征的重要性也会相当高。这是特征的属性导致的，那么在打乱标签输出的null importance中，具有这种问题的特征重要性还是会相对较高，因此便可以用null importance来对实际的importance做一个修正，得到真实的特征重要性。
<a name="OU2ec"></a>
# 代码实现
下面结合自己的理解看下代码实现：<br />首先我们需要一个函数用来计算importance, 并且实际的importance和null importance都能计算：
```python
def get_feature_importances(data, shuffle, seed=None):
    # Gather real features
    train_features = [f for f in data if f not in ['TARGET']]
    # Go over fold and keep track of CV score (train and valid) and feature importances

    # Shuffle target if required
    y = data['TARGET'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['TARGET'].copy().sample(frac=1.0)

        # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': 4
    }

    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=categorical_feats)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))

    return imp_df

```
这里实际上用的是lightgbm实现的随机森林模型，因为随机森林输出的特征重要性更稳定（多颗决策树的叠加），GBDT还需要考虑迭代次数过多而导致的过拟合问题，而随机森林不需要，因此用了随机森林。最终输出的importance有两个，一一个是split，代表这个特征被决策树选择用来分裂的次数，另外一个是gain，代表这个特征分裂时带来的增益总和，这里两种类型都输出，两种结果有点差异，最终自己选一种就行了。<br />然后接下来计算实际的importance和null importance：
```python
actual_imp_df = get_feature_importances(data=data, shuffle=False)

null_imp_df = pd.DataFrame()
nb_runs = 80
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=data, shuffle=True)
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

```
最后计算特征的分数：
```python
feature_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
    feature_scores.append((_f, split_score, gain_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
scores_df = scores_df.sort_values('split_score', ascending=False)

```
根据输出的score_df便可以得到每个特征的分数，也可以理解为它们真实的特征重要性。
<a name="iYWob"></a>
# 总结
Null Importance以打乱标签的方式来输出null importance，来对实际的importance进行修正，从而得到真实的特征重要性，和Boruta一样，是针对树模型的特征选择方法，不过Boruta是打乱特征，而Null Importance是打乱标签，因此Null Importance对于内存的占用和速度都会比Boruta要好。
