# NYCU Machine Learning 2024: HW2 Report

> Written By 313511068 練鈞揚
---

## Introduction

In this assignment, we are using LDA to classify the Iris data set, and using the ROC and AUC to analyze the model effect by penalty weight. Last, using the LDA to do multi-classification with one against one strategy.

## Experiment

### LDA in 2 classification

|    | Name                                  | Weight Vector   |   Bias | Acc         |
|---:|:--------------------------------------|:----------------|-------:|:------------|
|  0 | (Before) Pos:Versicolor,Neg:Virginica | [-2.09,-10.46]  |  28.1  | Acc :94.00% |
|  1 | (After) Pos:Versicolor,Neg:Virginica  | [-3.73,-7.85]   |  31.08 | Acc :94.00% |

> Average Acc : 94.00%

### LDA in 2 classification ROC AUC Graph

***Dataset: Using all feature***

![alt](./assets/part3/Using%20all%20feature.jpg)

|    | Model             |   AUR |
|---:|:------------------|------:|
|  0 | before_c1=1,c2=1  |  0.94 |
|  1 | after_c1=1,c2=1   |  0.94 |
|  2 | before_c1=1,c2=10 |  0.98 |
|  3 | after_c1=1,c2=10  |  0.94 |
|  4 | before_c1=10,c2=1 |  0.88 |
|  5 | after_c1=10,c2=1  |  1    |

***Dataset: Using 1,2 feature***

![alt](./assets/part3/Using%201,2%20feature.jpg)

|    | Model             |   AUR |
|---:|:------------------|------:|
|  0 | before_c1=1,c2=1  |  0.74 |
|  1 | after_c1=1,c2=1   |  0.7  |
|  2 | before_c1=1,c2=10 |  0.5  |
|  3 | after_c1=1,c2=10  |  0.56 |
|  4 | before_c1=10,c2=1 |  0.5  |
|  5 | after_c1=10,c2=1  |  0.62 |

***Dataset: Using 3,4 feature***

![alt](./assets/part3/Using%203,4%20feature.jpg)

|    | Model             |   AUR |
|---:|:------------------|------:|
|  0 | before_c1=1,c2=1  |  0.94 |
|  1 | after_c1=1,c2=1   |  0.94 |
|  2 | before_c1=1,c2=10 |  0.94 |
|  3 | after_c1=1,c2=10  |  0.9  |
|  4 | before_c1=10,c2=1 |  0.82 |
|  5 | after_c1=10,c2=1  |  0.94 |

### LDA in multi classification

|    | Name                                 | Weight Vector                                  | Bias               |Acc         |
|---:|:-------------------------------------|:-----------------------------------------------|:-------------------|:------------|
|  0 | (Before) Setosa,Versicolor,Virginica | [-19.67,-15.01],[-12.65,-34.96],[-2.09,-10.46] | 68.72,84.97,28.10  |Acc :96.00% |
|  1 | (After) Setosa,Versicolor,Virginica  | [-14.67,-17.11],[-34.15,-23.52],[-3.73,-7.85]  | 54.87,144.81,31.08 |Acc :96.00% |

> Average Acc : 96.00%

## Analysis

## Conclusion
