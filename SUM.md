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














