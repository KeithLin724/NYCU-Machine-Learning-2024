# NYCU Machine Learning 2024 : HW 5 Report

> Written By KYLiN
---

## Introduction

Implement two feature selection methods (Sequential Forward Selection and Fisher’s Criterion); compare the similarities and differences between Filter-based and Wrapper-based feature selection methods; and use the breast cancer dataset, along with the LDA classifier and 2-Fold CV, to complete the classification task, evaluating the classifier's performance using the balanced classification rate.

---

## Report on Feature Selection Methods: Sequential Forward Selection and Fisher’s Criterion

> Result in this [folder](./output/)

### 1. Which type of feature selection methods do Sequential Forward Selection and Fisher’s Criterion belong to among Filter-based and Wrapper-based methods?

Sequential Forward Selection (SFS) is a **Wrapper-based** feature selection method. It evaluates feature subsets by iteratively adding features and assessing the performance using a specific learning algorithm.

Fisher’s Criterion is a **Filter-based** feature selection method. It selects features based on statistical measures, independent of any learning algorithm, by maximizing the class separability.

### 2. Generally, what are the characteristics or advantages and disadvantages of Filter-based and Wrapper-based feature selection methods?

- **Filter-based Methods:**
  - *Characteristics:* Select features based on statistical measures without involving any learning algorithms.
  - *Advantages:*
    - Computationally efficient and fast.
    - Less prone to overfitting since they are not tailored to a specific model.
    - Can be used as a preprocessing step.
  - *Disadvantages:*
    - May select features that are not optimal for the specific learning algorithm.
    - Ignore feature dependencies and interactions.

- **Wrapper-based Methods:**
  - *Characteristics:* Use a specific learning algorithm to evaluate the performance of feature subsets.
  - *Advantages:*
    - Consider interactions between features and the learning algorithm, potentially leading to better performance.
    - Tailored feature selection for specific models.
  - *Disadvantages:*
    - Computationally intensive and slower due to repeated training.
    - Higher risk of overfitting, especially with small datasets.

### 3. From the results of this assignment, did you observe any phenomena consistent with your answer to the previous question?

Yes, the results align with the characteristics mentioned above:

#### Fisher’s Criterion (Filter-based)

![img](./output/fisher_criterion.png)

- **Best Balanced Classification Rate:** **0.9375** achieved with **18** features.
- **Average Accuracy:** **0.9368**.
- **Selected Features:**

    ```json
    [
        "worst concave points",
        "worst perimeter",
        "mean concave points",
        "worst radius",
        "mean perimeter",
        "worst area",
        "mean radius",
        "mean area",
        "mean concavity",
        "worst concavity",
        "mean compactness",
        "worst compactness",
        "radius error",
        "perimeter error",
        "area error",
        "worst texture",
        "worst smoothness",
        "worst symmetry"
    ]
    ```

Achieved a maximum balanced classification rate of approximately **0.937** with 18 features. It selected features based on statistical measures without considering the LDA classifier in the selection process. (click [here](./output/fisher_criterion.json) to view more)

#### Sequential Forward Selection (Wrapper-based)

![img](./output/sequential_forward_selection.png)

- **Best Balanced Classification Rate:** **0.9528** achieved with **13** features.
- **Average Accuracy:** **0.9544**.
- **Selected Features:**

    ```json
    [
        "worst concave points",
        "mean fractal dimension",
        "worst texture",
        "worst radius",
        "worst area",
        "worst smoothness",
        "mean radius",
        "worst fractal dimension",
        "worst symmetry",
        "area error",
        "worst perimeter",
        "mean area",
        "smoothness error"
    ]
    ```

Achieved a higher maximum balanced classification rate of approximately **0.953** with 13 features. It evaluated feature subsets using the LDA classifier, leading to better performance but likely requiring more computational resources. (click [here](./output/sequential_forward_selection.json) to view more)

This demonstrates that the Wrapper-based method provided better performance by considering the interaction between features and the learning algorithm, consistent with the advantages and disadvantages discussed.

### Analysis

![img](./output/All.png)

#### Performance

- **SFS** achieved a higher balanced classification rate and average accuracy compared to **Fisher’s Criterion**.
- **SFS** reached optimal performance with **fewer features** (13 features) than **Fisher’s Criterion** (18 features), indicating a more efficient feature set.

#### Feature Selection

- **Sequential Forward Selection (Wrapper-based Method):**
  - Evaluates feature subsets using the LDA classifier.
  - Considers interactions between features and the classifier.
  - May capture feature synergies that improve classification performance.

- **Fisher’s Criterion (Filter-based Method):**
  - Selects features based on statistical measures independently of the classifier.
  - Ranks features according to their individual discriminative power.
  - May not account for redundancy or interactions between features.

### Conclusion

The results demonstrate that **Sequential Forward Selection** outperforms **Fisher’s Criterion** in this classification task:

- **Higher Performance:** Achieved better classification metrics with fewer features.
- **Consistency with Method Characteristics:**
  - **Wrapper-based methods** like SFS can provide superior performance by tailoring the feature selection to the specific classifier but are more computationally intensive.
  - **Filter-based methods** like Fisher’s Criterion are faster and simpler but may not select the most optimal feature subset for the classifier.

These findings align with the general advantages and disadvantages of wrapper-based and filter-based feature selection methods.
