# WaikatoOU-COMP321
Parctical on Data mining
*summary of lecture slides*

# Introduction:

### Data mining
Identifying implicit, previously unknown, potentially useful patterns in data
* can be done interactively or automatically 
* can be descriptive or predictive

### Machine learning for data mining
programs that induce structural description from observations (examples)
* Supervised learning is based on labelled examples and used for predicting labels of new observations
* unsupervised learning is based on unlabelled data

### Classification rule:
Predicts value of a given attribute (classification of an example)
### Association rule:
Predicts value of arbitrary attribute (or combination)

-----------------------------
# Input:
## Concepts: structures that can be learned
* Aim: intelligible and operational concept description
*Concept description: output of learning scheme*
## instances: the individual, independent examples of a concept, possibly corrupted by noise
* Note: more complicated forms of input are possible*
## Attributes: measuring aspects of an instance
* We will focus on nominal and numeric ones
 
### Classification learning:
1. Supervised
2. Outcome is called the class of the example
3. Can measure success on test data
### Numeric prediction:
1. Supervised
2. measure success by comparing predictions to actual values in test data
### Association learning:
1. Unsupervised learning
2. Input normally consists of purely nominal attributes
3. Detecting associations between attributes
#### difference between classifications:
*-can predict any attributes values, not just the class  
-more association rules than classification rules*
### Clustering
1. unsupervised
2. Find groups of items that are similar

### Demoralization (generate a flat file)	
Strategy to improve the read performance of database

### Attributes
Each instance is described by a fixed predefined set of features
1. nominal attributes (values are labels or names)
2. Ordinal attributes (impose order on values)
3. Interval quantities (ordered and measured in fixed and equal unites)
4. ratio quantities(比例数据)

### ARFF fromat	
``` @relation weather
@attribute outlook {sunny, overcast, rainy}
@attribute temperature numeric
@attribute humidity numeric
@attribute windy {true, false}
@attribute play? {yes, no}
@data
sunny, 85, 85, false, no
sunny, 80, 90, true, no
overcast, 83, 86, false, yes
```

### Sparse data
*Eliminate the zero in the data set then make a position to indicate the no-zero value*
``` orginal
0, 26, 0, 0, 0 ,0, 63, 0, 0, 0, “class A”
0, 0, 0, 42, 0, 0, 0, 0, 0, 0, “class B”
```

```sparse 
{1 26, 6 63, 10 “class A”}
{3 42, 10 “class B”}
```

### Missing values
* Indicated using “?” character in ARFF format for unknown unrecorded irrelecant
* “Missing” could be an additional value.
### Unbalanced data
* Classification learning can suffer from data imbalance
* Solution: assign a cost to different type misclassification

--------------------------------
# Output
## Decision tables
The simplest way of representing the output of learning (reduced form of the output)

## Linear regression 
* Give the distribution of instance by a string
* Linear classification 
* Use the line to divide different classes

## Decision tree
* Divide-and-conquer method to process the training data to decision tree
* Internal nodes in the tree test attributes
* Leaves assign classification or probability distribution to instances
* To make prediction, instance with unknown class label is routed down the tree

## Nominal and numeric attributes
### Nominal attribute:
One branch for every value
### Numeric attribute:
* Test the value is greater or less than constant
* Different possibility:three-way split or multi-way split

## Missing values
* Missing is important (should be a separate value)
* Missing is not important   
`  solution A: assign to popular branch `
`  solution B: split instance into pieces,with one piece per branch extending from node`

## Classification rules
* Alternative decision trees
1. Antecedent (pre-condition) a series of tests ANDed together
2. Consequent (conclusion) class or probability distribution assigned by rule ORed together

## From trees to rules
* One rule for each leaf:
* **antecedent** contain a condition for every node on the path from the root to the leaf
* **consequent** is class assigned by the leaf
* Rules could be unnecessarily complex   
apply pruning to remove redundant rules

## From rule to trees
* Symmetry needs to be broken
* Corresponding tree contains replicated subtree

## Nuggets of knowledge
Two ways of executing a rule set 
* Ordered set of rules-order is important for interpretation
* Unordered set of rules-rules may overlap and lead to different conclusions for the same instance  
**Special case Boolean class does not affected by order** 

## Association rules
* Can predict any attribute and combinations of arrtibutes
* -not intended to be used together as a set
* Problem: exist large amount of association 
`Solution: only for high support and high confidence`

## Support and confidence of a rule
* Support: number of correct instances
* Confidence: number of correct predictions as proportion of all instances

### Instance-Based Representation

Lazy learning, AKA Rote Learning

-   Training instance are searched for instances which are most closely resemble test instance.
    
-   The instances themselves represent the knowledge.
    

Methods: `Nearest-Neighbor`, `K-Nearest-Neighbor`

### 基于实例的表达

懒惰学习，又称死记硬背法学习

-   训练的实例由那些最接近于测试实例的实例构成。
    
-   实例本身就表达了知识。
    

方法：`Nearest-Neighbor`, `K-Nearest-Neighbor`

### Distance Function

-   Numeric: Euclidean distance with normalization
    
-   Nominal: 1 - different, 0 - equal
    

### 距离函数

-   数值：规范的欧氏距离
    
-   名目：1 - 不同， 0 - 相同
    

## Inferring Simple Rules


### Inferring simple rules

simple algorithms often work very well in practical data mining problems

简单算法通常在实际数据挖掘问题中运行良好

### There are many kinds of simple structure 
* one attribute does all the work 
* all attributes contribute equally & independently
* a weighted linear combination might do
* a decision tree may exhibit pertinent interactions 
* some simple logical rules may be sufficient
* success of method depends on the domain
### 有很多种简单的结构
* 一个属性完成所有工作
* 所有属性均等且独立地贡献
*  加权线性组合可能会做
* 决策树可以展示相关的交互
* 一些简单的逻辑规则可能就足够了
* 方法的成功取决于域

### Inferring Rudimentary Rules 推断基本规则

Learns a 1-level decision tree(rules that all test one particular attribute) 学习1级决策树（所有测试一个特定属性的规则）

* Basic version
    * one branch for each value 每个值都有一个分支
    * each branch assigns most frequent class 每个分支分配最频繁的类
    * error rate:proportion of instances that don't belong to the majority class for their corresponding branch 错误率：不属于其对应分支的多数类的实例的比例
    * choose attribute with lowest error rate 选择具有最低错误率的属性

(assumes nominal attributes 假定名义属性)

### Dealing with numeric attributes 处理数字属性

* Solution: discretize numeric attributes 
* Divide each attribute's range into intervals
* Sort instances according to attribute's values
* Place breakpoints where class changes(majority class)
* This minimizes the total error 
* 解决方案：离散数字属性
* 将每个属性的范围划分为间隔
* 根据属性的值对实例进行排序
* 在类更改的地方放置断点（多数类）
* 这最小化了总误差

### Over-fitting Problem 过度拟合问题

* This procedure is very sensitive to noise 
* One instance with an incorrect class label will probably produce a separate interval 
* Time stamp attribute will have zero errors 
* 这个处理对噪音非常敏感
* 一个具有不正确的类标签的实例可能会产生一个单独的间隔
* 时间戳属性将没有错误

Simple solution:
enforce minimum number of instances in majority class per interval
每个间隔强制执行多数类中的最小实例数

### Others

1R's simple rules performed not much worse than much more complex decision tree

1R的简单规则执行并不比更复杂的决策树差

### oneR 算法介绍

OneR是One Rule的意思，即一个规则，只看某事物的一个特征，然后来预测该事物的类别。

比如，现在给出一条数据，该数据描述了某一日天气的三个特征：紫外线强度，温度，湿度。要根据这三个特征来预测该当天的天气是晴，多云或者下雨。以日常思维进行判断的话，一般可以进行如下预测：如果紫外线强度为高，温度为30度，湿度为中，那么我们预测当日天气为晴天。这个预测是根据三个数据的特征的出来的。但是oneR算法则是希望找出一个准确率最高的attribute，比如紫外线强度来预测天气情况。比如，紫外线强度弱就是雨天（只考虑一般情况，不考虑高原之类的特殊环境）。

**具体可以查看下面的链接有非常详细的介绍和实例**

[https://blog.csdn.net/baidu_25555389/article/details/73379036](https://blog.csdn.net/baidu_25555389/article/details/73379036)

### Pseudo-code 伪代码
![pseudo.jpg](https://upload-images.jianshu.io/upload_images/13576645-afe51755cd676a34.JPG)
Missing 被视为单独的属性值

## Practical Data Mining

### Naive statistical modeling

* use all the attributes, not just one, to derive classification
* Based on describing the probability distribution of the data in each class of instances
  * Information from different classes can then be combined into classification using Bayes' rule
* Called “naive” because attributes are assumed to be statistically independent for each class
* Independence assumption is almost never correct but works surprisingly well in practice

### 简单统计模型

* 使用所有的属性，并不仅仅用一个属性来分来。
* 基于描述每类实例中数据的概率分布
  * 然后可以使用bayes rule将来自不同类别的信息组合成分类
* 被叫做“天真”，因为假设属性在每个类中都具有统计独立性
* 独立性假设几乎从不正确，但在实践中效果出奇的好
 
### Bayes’ rule

* Probability of event H given evidence E
  * Pr[H∣E]=Pr[E∣H]Pr[H]/Pr[E]

* A priori probability of H : Pr[H]
  * Probability of event before evidence has been seen

* A posteriori probability of H : Pr[H∣E]
  * Probability of event after evidence has been seen 

### bayes 规则

* Pr[H∣E] 概率公式
* Pr[H] ： H的先验概率
  * 在看到证据之前发生事件的可能性
* Pr[H∣E] : H的后验概率
  * 在看到证据之后发生事件的可能性

### Naive Bayes for classification

* what’s the probability of the class given an instance? 
  * Evidence E = instance described by attributes E1... En
  * Event H = class value for instance
* To apply Bayes' rule, we need to estimate Pr[E|H]
* Naïve assumption: attributes that are independent when considering a particular class value H
* Then Pr[E∣H]=Pr[E1∣H]Pr[E2∣H]...Pr[En∣H] and Bayes' rule becomes:
  Pr[H∣E]=Pr[E1∣H]Pr[E2∣H]...Pr[En∣H]Pr[H]/ Pr[E]

### Naive Bayes 分类

* class given an instance 的概率
  * 证据E = E1..En 描述的实例
  * 事件H = 对于instance class的值
* 要应用贝叶斯规则，我们需要估计Pr [E | H]
* 天真的假设：在考虑特定的类值H时，这些属性是独立的

### zero-frequency problem

* Probability will be zero Pr[Humidity = High|yes]=0
* A posteriori probability will also be zero: Pr[yes|E]=0
* Remedy ： add 1 to the count for every attribute value-
class combination
* Result : probabilities will never be zero(also: stabilizes probability estimates)

### 零频率问题

* 概率为0
* 后验概率也将为零
* 补救措施：为每个属性值的计数加1
* 结果：概率永远不会为零（同样：稳定概率估计）

### Missing values

* Training: instance is not included in frequency count for attribute value-class combination
* Classification: attribute is omitted from calculation

### 丢失数据

*培训：实例不包含在属性值类组合的频率计数中
*分类：计算中省略属性

### Multinomial naïve Bayes 

* Version of naïve Bayes used for document classification by applying the bag of words model
* n : number of times word i occurs in bag
* P : probability of obtaining word i when picking a word at random from documents in class H
* Probability of observing a particular bag of words E given class H (based on 
multinomial distribution):Pr[E∣H]≈ (N!/ ∏(i=1,k)ni!) * ∏(i=1,k)pi!
* The N words in the bag are assumed to have been picked in independent
trials based on the given probabilities

###多项式幼稚贝叶斯

* 通过应用词袋模型用于文档分类的朴素贝叶斯版本
* n：单词i出现在包中的次数
* P：从H类文档中随机选取单词时获得单词i的概率
* 在给定H级的情况下观察特定词袋E的可能性（基于多项分布）：Pr [E | H]≈（N！/Π（i = 1，k）ni！）*Π（i = 1，k）pi！
* 假设包中的N个单词是独立挑选的基于给定概率的试验

### discussion

* Naive Bayes works surprisingly well even if independence assumption is clearly violated
* Why? Because accurate classification doesn’t necessarily require accurate probability estimates
* However: adding too many redundant attributes will cause problems (e.g., identical attributes)

###讨论

* 即使明显违反了独立性假设，朴素贝叶斯的工作也出奇地好
* 为什么？ 因为准确的分类不一定需要准确的概率估计
* 然而：添加太多多余属性会导致问题（例如，相同的属性）




## Decision Tree Learning
1. Constructing decision trees
    * Strategy: top down
    * Recursive divide-and-conquer fashion
        - First: select attribute for root node 
            Create branch for each possible attribute value
        - Then: split instances into subsets
            One for each branch extending from the node
        - Finally: repeat recursively for each branch, using only instances that reach the branch
    * Stop if all instances have the same class
    
    * 策略: 从上到下
    * 递归各个击破的方式
        - 首先: 选择一个根的属性，为每个可能的属性创建分支
        - 然后: 将实例分解为子集，每个分支从节点扩展一个分支
        - 最后: 只使用到达该分支的实例，对每个分支递归地重复。
    * 如果所有的实例拥有相同的类则停止。
2. Criterion for attribute selection
    * Which is the best attribute?
        - Want to get the smallest tree possible
        - Heuristic: choose the attribute that produces the “purest” nodes
    * Popular criterion: information gain
        - Information gain increases with the average purity of the subsets
    * Strategy: choose attribute that gives greatest information gain

    * 那一个是最好的属性?
        - 想要得到尽可能最小的树
        - 启发式: 选择可以产生“最纯粹”的节点的属性
    * 流行的标准: 信息增益值
        - 信息增益值随着子集的平均纯度而增加
    * 策略: 选择获得最大信息收益值的属性
3. Computing information
    * Can measure information in bits
      - Given a probability distribution, the average information required to specify an event is the distribution’s entropy
      - Entropy gives the expected information in bits (can involve fractions of bits)
    * Formula for computing the entropy:
        entropy(p1,p2,...,pn)= −p1logp1 − p2logp2... −pnlogpn

    * 可以以比特为计量单位的方式测量信息
        - 用给出的概率分布来计算出来的分布的熵是指定事件所需的平均信息量
        - 熵给出了以比特为计量单位的信息 (可以包含比特的分数)
    * 用来计算熵的公式:
        熵(p1,p2,...,pn)= −p1logp1 − p2logp2... −pnlogpn

    * 例子：Outlook = Sunny :
            info([2,3])=entropy(2/5, 3/5)=−2/5log(2/5) − 3/5log(3/5) = 0.971bits
4. Final decision tree
    * Note: not all leaves need to be pure; sometimes identical instances have different classes
         → Splitting stops when data can’t be split any further
    * May enforce minimum leaf size to avoid overfitting
    * Can predict class labels or class probabilities (based on relative frequencies of classes at leaf)

    * 注意: 不是所有的节点需要变成“最纯粹的”; 有时相同的实例具有不同的类
         → 当数据不能被进一步分割时，分割停止
    * 可以强制最小叶径以避免过度拟合
    * 基于类的节点的频率可以预测类的标签和类的概率
    
5. Highly-branching attributes
* Problematic: attributes with a large number of values (extreme case: ID code)
    * Subsets are more likely to be pure if there is a large number of values
        - Information gain is biased towards choosing attributes with a large number of values
        - This may result in overfitting (selection of an attribute that is non-optimal for prediction)
    * Another problem: fragmentation
    
    * 问题:具有大量值的属性(极端情况:ID代码)
    * 如果有大量的值，子集更可能是纯的
        - 信息增益倾向于选择具有大量值的属性
        - 这可能导致过度拟合(选择不适合预测的属性)
    * 另一个问题:分裂

6. Gain ratio
    * Gain ratio: a modification of the information gain that attempts to correct its bias
    * Gain ratio takes number and size of branches into account when choosing an attribute
        - It corrects the information gain by taking the intrinsic information of a split into account
    * Intrinsic information: entropy of distribution of instances into branches
        - i.e., how much info do we need to tell which branch an instance belongs to
        
    * 增益比:对信息增益的修正，以纠正其偏差
    * 在选择属性时，增益比率会考虑到分支的数量和大小
        - 它通过考虑分割的内在信息来修正信息增益
    * 内在信息:实例分布到分支的熵
        - i.e., 我们需要多少信息才能知道一个实例属于哪个分支
    
7. Computing the gain ratio
    * Example: intrinsic information for ID code
        ratio:info([1,1,...,1])=14×(−1/14×log(1/14))=3.807bits
    * Worth of attribute is assumed to decrease as intrinsic information gets larger
    * Definition of gain ratio: gain_ratio(attribute) = gain(attribute) /i ntrinsic_info(attribute)
    * Example: gain_ratio(ID code) = 0.940bits ／ 3.807bits = 0.246
    
    * 例子: 内部信息 for ID code
        ratio:info([1,1,...,1])=14×(−1/14×log(1/14))=3.807bits
    * 假设属性值随着内在信息的增大而减小
    * 获得率的定义: gain_ratio(attribute) = gain(attribute) /i ntrinsic_info(attribute)
    * 例子: gain_ratio(ID code) = 0.940bits ／ 3.807bits = 0.246
    
8. Discussion
    * Top-down induction of decision trees: ID3 by R. Quinlan
        - Gain ratio just one modification of this basic algorithm
        - Advanced version: C4.5 (J4.8 in WEKA)
    * Chooses best split point on numeric attributes by maximizing information gain (cf. discretization in 1R)
    * Deals with missing attribute value by splitting instance into pieces and combining predictions for pieces
    * Includes tree pruning to combat overfitting: subtrees and nodes are pruned if this reduces estimated error
    * Alternative approach: CART (SimpleCART in WEKA)
        - Differs in how splits are selected, missing values are dealt with, and errors for pruning are estimated
        - Error estimation using cross-validation, not based on training data → less likely to overfit, but slower
        
    * 决策树的自上而下归纳法: ID3 by R. Quinlan
        - 增益比只是这个基本算法的一个修改
        - 高级版本:C4.5 (WEKA中的J4.8): C4.5 (J4.8 in WEKA)
    * 通过最大化信息增益来选择数值属性的最佳分割点(如1R中的离散化)
    *通过最大化信息增益来选择数值属性的最佳分割点(如1R中的离散化)
    *通过将实例分割成块并结合对块的预测来处理丢失的属性值
    *包括树木修剪，以对抗过度拟合:如果子树和节点被修剪，如果这减少估计误差
    *替代方法:CART (WEKA中的SimpleCART)
        -不同的分裂如何被选择，缺失的值被处理，和修剪的错误被估计使用交叉验证,误差估计,不是基于训练数据→overfit较少,但速度较慢
        
9. Beyond individual trees
    * Trees are popular because they can be generated very quickly; also, they can provide insight
    * But: generally not the best approach when goal is to maximize predictive performance
        ̵ One reason: trees are quite unstable (i.e., small changes in the training data can yield different tree)
    * Better: using ensemble of trees
        ̵ Idea: generate collection of different trees, let them vote on a classification (or average class probabilities)
        ̵ Example methods: bagging (change input data), randomization (semi-random split selection),
            boosting (build trees sequentially, focus on mistakes)

    *    树很受欢迎，因为它们可以很快生成;此外，它们还能提供洞察力
    * 但是:当目标是最大化预测性能时，通常不是最好的方法
        ̵ 原因之一:树很不稳定(即。，对训练数据的小修改可以产生不同的树)
    * 更好的方法:使用树木的整体效果
        ̵ 想法:生成集合不同的树木,让他们投票表决一个分类(或平均类概率)
        ̵ 示例方法:装袋(改变输入数据),随机化(半随机分割选择),
    促进(按顺序构建树，关注错误)

## Mining Association Rules
Association rule mining is an unsupervised learning task: goal is knowledge discovery 
关联规则挖掘为非监督，目标为发现knowledge.对比监督learning, 无监督learning是否成功很难评估。

Main criterion for evaluating rule mining algorithms: computational efficiency
计算效率是挖掘算法的主要标准。

### Goal
set of association rules that have high support and high confidence
找到高支持度和高置信度的关联规则集合。

### Possible method for finding association rules
Use separate-and-conquer rule learning approach 变治法
Treat every possible combination of attribute values as a separate class, run rule learner repeatedly
通过变治法，或将每个可能的属性值视为一个相同的类并反复运行。

### Two problems
1. Computational complexity 
2. Resulting number of rules (which would have to be pruned on the basis of support and confidence)
产生两个问题：计算复杂度与产生的规则数量
替代方案： 直接寻找高支持度的规则

### Frequent item sets
Support：number of instances correctly covered by association rule 
The same as the number of instances covered by all tests in the rule (antecedent and consequent!)
支持度：实例个数由关联规则正确地覆盖
相同地，实例的个数被所有规则中的测试覆盖掉。(包括先决条件和结果)

### 关联规则中的术语
Item: one test corresponding to an attribute-value pair  对应于一个属性值的测试
- Item set: all items occurring in a rule 规则中出现的所有项目

### Goal of rule mining
All rules that exceed pre-defined support level and pre-defined confidence level
所有超出预先定义支持度和置信度的规则
Two-step process that is generally used for this: 
1. Generate all item sets that satisfy specified minimum support level (so-called frequent item sets) 
2. From these frequent item sets, generate all rules that satisfy minimum confidence level
生成所有满足最低支持度的项目集，从这些项目集中在生成所有满足最低自信度的规则。


### How can we efficiently find all frequent item sets? 
### 怎样找到所有的频繁项集
The number of possible item sets grows exponentially with the number of attributes.
可能的项目集数目可能随着属性数目的增长呈指数级增长。
#### Finding one-item sets is easy
one scan through the data is sufficient to count how often they occur.

Idea: use one-item sets to generate two-item sets, two-item sets to generate three-item sets, … − If (A B) is frequent item set, then (A) and (B) have to be frequent item sets as well! − In general: if X is frequent k-item set, then all (k-1)- item subsets of X are also frequent 
基本的方法是一个单项项目集生成双项项目集，双项项目集合生成三项项目集 以此类推
概括的来讲，如果X 是一个N 项频繁集，则 所有的n-1 项项目集都是频繁集

Idea: compute k-item set by merging (k-1)-item sets
通过合并 n-1 项集来计算n项集
所有k-1 项必须为频繁项

Confidence = support of full item set divided by support of Antecedent
置信度 = 对所有项集的支持除以先决条件的支持
*Support of antecedent can be obtained from hash table that contains all frequent item sets
*先决条件可以从包含所有频繁项集的哈希表中获得

Brute-force method is (2N-1), where N is the number of items in the frequent item set concerned
暴力方法是（2N-1），其中N是频繁项目集中的数字
Better way: building (c + 1)-consequent rules from consequent ones 
建立一个个c+1的后续规则

Observation: (c + 1)-consequent rule can only hold if all corresponding c-consequent rules also hold
观察：仅当所有对应的c-后续规则成立时，(c+1)后续规则也就成立了。
Reason: confidence cannot decrease by moving item set from consequent to antecedent. Algorithm similar to procedure for large item sets.
理由：置信度不会因将项集从结论向先决条件移动而降低。算法可能生成大量的项集。

### Algorithm for mining item sets and rules is called Apriori
### 调用挖掘项集和规则的算法 先验
* Practical issue: number of frequent item sets explodes when minimum support is too low 
* WEKA implementation generates a certain user specified number of rules
    * Does this by running the algorithm repeatedly with incrementally reduced min. support values 
* Confidence is not necessarily the best measure for finding interesting rules − Example: milk occurs in most supermarket transactions, not useful if we can predict this with high confidence − Other measures have been devised (e.g., lift)
* 支持度低的时候频繁项目集的数量会特别多
* 不停的运算减少支持度
* 置信度并没有必要是对于找到有意思的规则的最佳验算。















