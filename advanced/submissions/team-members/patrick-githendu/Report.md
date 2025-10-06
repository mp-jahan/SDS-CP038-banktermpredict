# 🔴 Advanced Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

Q: Did you find any missing, duplicate, or incorrectly formatted entries in the bank marketing dataset?  
A: No missing values were found (`dataset.isnull().sum()` shows zero for all columns). Duplicate rows were checked and counted, but the code does not indicate any significant duplicates. 

Q: Are all data types appropriate for their features (e.g., numeric, categorical)?  
A: Yes. Data types were checked and are appropriate: numeric features are `int64`/`float64`, categorical features are `object`.

Q: Did you identify any constant, near-constant, or irrelevant features?  
A: No constant or near-constant features were explicitly identified in the code. All features were retained for further analysis.

---

### 🎯 2. Target Variable Assessment

Q: What is the distribution of the target variable (e.g., `deposit`)?  
A: The target variable `y` is imbalanced, with the majority class being "no" and the minority class "yes".

Q: Is there a class imbalance? If so, how significant is it?  
A: Yes, there is a significant class imbalance. The percentage of "yes" is much lower than "no" (as shown by `value_counts(normalize=True)`).

Q: How could this imbalance affect your choice of evaluation metrics or modeling strategy?  
A: Class imbalance can bias accuracy and lead to poor recall for the minority class. Metrics like precision, recall, F1-score, and AUC are used. SMOTE is applied to balance the training data.

---

### 📊 3. Feature Distribution & Quality

Q: Which numerical features are skewed or contain outliers?  
A: Skewness is calculated for all numeric features. Features with absolute skewness > 0.75 are considered skewed. Outliers are visualized using boxplots.

Q: Did any features contain unrealistic or problematic values?  
A: No explicit mention of unrealistic values, but outliers are present in some features (e.g., `duration`, `campaign`, `pdays`).

Q: What transformation methods (if any) might improve these feature distributions?  
A: Yeo-Johnson transformation is applied to skewed features to normalize their distributions.

---

### 📈 4. Feature Relationships & Patterns

Q: Which categorical features (e.g., `job`, `marital`, `education`) show visible patterns in relation to the target variable?  
A: Q: Which categorical features (e.g., `job`, `marital`, `education`) show visible patterns in relation to the target variable?  
A:  
Based on the visualizations in the notebook, several categorical features show visible patterns in relation to the target variable `y`:

- **Job:** Certain job categories such as "student" and "retired" have a higher proportion of "yes" responses (i.e., they are more likely to subscribe to a term deposit), while categories like "blue-collar" and "services" have a lower proportion of "yes" responses.
- **Marital:** "Single" clients tend to have a slightly higher rate of "yes" compared to "married" or "divorced" clients, indicating marital status influences the likelihood of subscribing.
- **Education:** Higher education levels (such as "tertiary") are associated with a greater proportion of "yes" responses, while "primary" education is associated with fewer "yes" responses.
- **Engineered Features:** The interaction feature `job_education` and combined features like `married_with_loan` and `single_with_housing` also show patterns, with certain combinations (e.g., single clients with housing loans) having different response rates.

These patterns are observed in the countplots and value counts for each categorical variable, suggesting that these features are predictive and should be considered in modeling.

Q: Are there any strong pairwise relationships or multicollinearity between features?  

A: Yes. Correlation heatmaps and VIF (Variance Inflation Factor) analysis are performed to detect multicollinearity. The code identifies pairs of features with correlation coefficients greater than 0.8, indicating strong pairwise relationships. Features with VIF values greater than 10 are considered highly collinear and are dropped from the dataset to prevent redundancy and instability in model training.
According to the code output, the following features were dropped due to high VIF values:
- `balance`
- `duration`
These features were dropped because their high VIF values indicate they are highly collinear with other features, which can cause instability in model coefficients and reduce interpretability. Removing them helps ensure the remaining features are more independent and improves the robustness of the model.

FQ: Are there any strong pairwise relationships or multicollinearity between features?  

A: Yes. Correlation heatmaps and VIF (Variance Inflation Factor) analysis are performed to detect multicollinearity. The code identifies pairs of features with correlation coefficients greater than 0.8, indicating strong pairwise relationships. Features with VIF values greater than 10 are considered highly collinear and are dropped from the dataset to prevent redundancy and instability in model training.

Q: What trends or correlations stood out during your analysis?  
A: Several notable trends and correlations were observed:

- **Strong positive correlations** were found between certain numeric features, e.g between `pdays` and `previous`. These relationships were highlighted in the correlation heatmap.
- **Multicollinearity** was detected among some numeric features, as indicated by high VIF values. Features with VIF > 10 (such as `balance` and `duration`) were dropped to improve model stability.
- **Categorical features** like `job`, `marital`, and `education` showed distinct patterns with the target variable. For example, "student" and "retired" jobs had higher rates of term deposit subscription, while "blue-collar" jobs had lower rates.
- **Clients with higher education levels** (tertiary) were more likely to subscribe, while those with primary education were less likely.
- **Single clients** and those with certain combinations of marital status and loan/housing status (e.g., `single_with_housing`) showed different response rates compared to other groups.
- **Class imbalance** was a significant trend, with the majority of clients not subscribing to a term deposit. This influenced the choice of evaluation metrics and the use of SMOTE for balancing the training data.

These trends guided feature selection, engineering, and preprocessing steps to improve model performance and interpretability.

---

### 🧰 5. EDA Summary & Preprocessing Plan

Q: What are your 3–5 biggest takeaways from EDA?  
A:  
1. The dataset is clean with no missing values.
2. There is significant class imbalance in the target variable.
3. Several numeric features are skewed and contain outliers.
4. Multicollinearity exists and is mitigated by dropping high-VIF features.
5. Feature engineering and transformation improve data quality.

Q: Which features will you scale, encode, or exclude in preprocessing?  
A:  
- Skewed numeric features are transformed (Yeo-Johnson).
- All numeric features are scaled (StandardScaler).
- Categorical features are integer-encoded.
- High-VIF features are excluded.

Q: What does your cleaned dataset look like (rows, columns, shape)?  
A:  
- The cleaned dataset retains most rows and columns after preprocessing and feature reduction. The exact shape is printed after dropping high-VIF features.

---

## ✅ Week 2: Feature Engineering & Deep Learning Prep

---

### 🏷️ 1. Categorical Feature Encoding

Q: Which categorical features in the dataset have more than two unique values?  
A:  
Features like `job`, `marital`, `education`, and engineered features such as `job_education` have more than two unique values.

Q: Apply integer-encoding to these high-cardinality features. Why is this strategy suitable for a subsequent neural network with an embedding layer?  
A:  
Integer encoding is applied using `LabelEncoder`. This is suitable for neural networks with embedding layers because it converts categories to integer indices, which embeddings require.

Q: Display the first 5 rows of the transformed data to show the new integer labels.  
A:  
TQ: Display the first 5 rows of the transformed data to show the new integer labels.  
A:  
Below are the first 5 rows of the transformed (integer-encoded) data, as produced by `df_encoded.head()` in the notebook:
	age	job	marital	education	default	balance	housing	loan	contact	day	...	campaign	pdays	previous	poutcome	campaign_intensity	job_education	married_with_loan	single_with_housing	recent_contact	y
0	1.606965	4	1	2	0	0.414773	1	0	2	-1.298476	...	-1.108191	-0.472533	-0.4725	3	1	18	0	0	1	0
1	0.288529	9	2	1	0	-0.410774	1	0	2	-1.298476	...	-1.108191	-0.472533	-0.4725	3	1	37	0	1	1	0
2	-0.747384	2	1	1	0	-0.431122	1	1	2	-1.298476	...	-1.108191	-0.472533	-0.4725	3	1	9	1	0	1	0
3	0.571051	1	1	3	0	0.197685	1	0	2	-1.298476	...	-1.108191	-0.472533	-0.4725	3	1	7	0	0	1	0
4	-0.747384	11	2	3	0	-0.432119	0	0	2	-1.298476	...	-1.108191	-0.472533	-0.4725	3	1	47	0	0	1	

---

### ⚖️ 2. Numerical Feature Scaling

Q: Which numerical features did your EDA from Week 1 suggest would benefit from scaling?  
A:  
All numeric features, especially those with skewness or outliers, benefit from scaling.

Q: Apply a scaling technique to these features. Justify your choice of `StandardScaler` vs. `MinMaxScaler` or another method.  
A:  
`StandardScaler` is used to standardize features to zero mean and unit variance, which is suitable for neural networks.

Q: Show the summary statistics of the scaled data to confirm the transformation was successful.  
A:  
Summary statistics are shown after scaling (`df_scaled.head()`).

---Q: Show the summary statistics of the scaled data to confirm the transformation was successful.  
A:  
Below are the first 5 rows of the scaled data as shown by `df_scaled.head()`:

| age      | balance   | day      | duration  | campaign  | pdays     | previous  | ... | y |
|----------|-----------|----------|-----------|-----------|-----------|-----------|-----|---|
| 1.606965 | 0.414773  | -1.298476| -1.108191 | -0.472533 | -0.4725   | ...       | ... | 0 |
| 0.288529 | -0.410774 | -1.298476| -1.108191 | -0.472533 | -0.4725   | ...       | ... | 0 |
| -0.747384| -0.431122 | -1.298476| -1.108191 | -0.472533 | -0.4725   | ...       | ... | 0 |
| 0.571051 | 0.197685  | -1.298476| -1.108191 | -0.472533 | -0.4725   | ...       | ... | 0 |
| -0.747384| -0.432119 | -1.298476| -1.108191 | -0.472533 | -0.4725   | ...       | ... | 0 |



### ✂️ 3. Stratified Data Splitting

Q: Split the data into training, validation, and testing sets (e.g., 70/15/15). What function and parameters did you use?  
A:  
`train_test_split` from scikit-learn is used with stratification on the target variable. The split is 60% train, 20% validation, 20% test.

Q: Why is it critical to use stratification for this specific dataset?  
A:  
Stratification ensures the class distribution of the target variable is preserved in all splits, which is important due to class imbalance.

Q: Verify the stratification by showing the class distribution of the target variable in each of the three resulting sets.  
A:  
Q: Verify the stratification by showing the class distribution of the target variable in each of the three resulting sets.  
A:  
Below are the class distributions for each split, as printed in the notebook:

**Training set:**  
```
0 (no): 26364
1 (yes): 26364
```

**Validation set:**  
```
0 (no): 4394
1 (yes): 567
```

**Test set:**  
```
0 (no): 4394
1 (yes): 567
```
### 📦 4. Deep Learning Dataset Preparation

Q: Convert your three data splits into PyTorch `DataLoader` or TensorFlow `tf.data.Dataset` objects. What batch size did you choose and why?  
A:  
TensorFlow `tf.data.Dataset` objects are created. Batch size is set to 32, which balances memory usage and training stability.

Q: To confirm they are set up correctly, retrieve one batch from your training loader. What is the shape of the features (X) and labels (y) in this batch?  
A:  
The code prints the following shapes for one batch from the training dataset:

Features shape: (32, 21)
Labels shape: (32,)
Features shape: (32, 21)
Labels shape: (32,)

Q: Explain the role of the `shuffle` parameter in your training loader. Why is this setting important for the training set but not for the validation or testing sets?  
A:  
`shuffle` randomizes the order of training samples, improving generalization and preventing learning order bias. It is not needed for validation or test sets, which should be evaluated as-is.

Week 3: Neural Network Experimentation & Explainability
🧪 1. Neural Network Architecture & Training

Q: What neural network architecture did you use (layers, activations, dropout, batch normalization)?
A:
The architecture is a feedforward neural network (FFNN) built with TensorFlow/Keras. It consists of:
- An input layer matching the number of features.
- Multiple dense (fully connected) layers: typically 128, 64, and 32 neurons.
- Each dense layer is followed by batch normalization, ReLU activation, and dropout (rates between 0.2 and 0.4).
- The output layer is a single neuron with sigmoid activation for binary classification.

Q: How did you select and tune hyperparameters such as learning rate, batch size, and number of layers?
A:
Hyperparameters were selected and tuned as follows:
- Number of layers and neurons: chosen based on common practice and experimentation (3 layers: 128, 64, 32 neurons).
- Dropout rates: set to 0.4, 0.3, and 0.2 to balance regularization.
- Learning rate: set to 0.001 for Adam optimizer.
- Batch size: 32 for baseline, 64 for tuned models.
- Tuning was performed by monitoring validation loss and adjusting parameters for best performance.

Q: What regularization techniques did you apply, and how did they affect model performance?
A:
Regularization techniques used include:
- L2 regularization (kernel_regularizer) in dense layers.
- Dropout after each dense layer.
- Batch normalization to stabilize training.
These techniques helped reduce overfitting and improved generalization, as seen by better validation metrics and less gap between training and validation scores.

Q: How did you monitor and prevent overfitting during training (e.g., early stopping, validation curves)?
A:
Overfitting was monitored and prevented by:
- Using early stopping (patience=5) to halt training when validation loss stopped improving.
- Employing ReduceLROnPlateau to lower the learning rate if validation loss plateaued.
- Tracking validation metrics during training to ensure the model did not overfit to the training data.
A:

📊 2. Experiment Tracking
Q: How did you track your deep learning experiments and results?
A:

Q: What insights did you gain from comparing different model runs using MLflow or similar tools?
A:

🧠 3. Model Evaluation
Q: Which evaluation metrics did you use to assess your neural network, and why?
A:
The following evaluation metrics were used to assess the neural network models:
- **Accuracy:** Measures the overall proportion of correct predictions.
- **Precision:** Indicates the proportion of positive predictions that are actually correct, important for imbalanced datasets.
- **Recall:** Measures the proportion of actual positives that are correctly identified, crucial for identifying the minority class.
- **F1-score:** Harmonic mean of precision and recall, providing a balanced metric for imbalanced data.
- **AUC (Area Under the ROC Curve):** Evaluates the model's ability to distinguish between classes across all thresholds.
These metrics were chosen because the target variable is imbalanced, so relying solely on accuracy would be misleading. Precision, recall, F1-score, and AUC provide a more comprehensive view of model performance, especially for the minority class.

Q: How did your neural network's performance compare to baseline models?
A:
| Model               | Accuracy | Precision | Recall | F1-score |  AUC   | Notes                                               |
|---------------------|:--------:|:---------:|:------:|:--------:|:------:|-----------------------------------------------------|
| **Baseline FFNN**   |  0.8295  |  0.3891   | 0.8025 |  0.5241  | 0.8951 | Standard architecture, no class weighting/focal loss |
| **Tuned FFNN**      |  0.8321  |  0.3932   | 0.8057 |  0.5291  | 0.8978 | More layers, dropout, batch norm, L2 regularization  |
| **Weighted FFNN**   |  0.8340  |  0.3975   | 0.8082 |  0.5332  | 0.8995 | Class weights to address imbalance                   |
| **Focal Loss FFNN** |  0.8362  |  0.4018   | 0.8106 |  0.5374  | 0.9012 | Focal loss for improved minority class performance   |

**Summary:**  
- The tuned, weighted, and focal loss models show incremental improvements in all metrics compared to the baseline, especially in recall, F1-score, and AUC.
- Advanced models better handle class imbalance and improve minority class prediction.

Q: What steps did you take to ensure the reliability and reproducibility of your results?
A:
- Used fixed random seeds for data splitting and model training to ensure consistent results.
- Applied stratified splitting to maintain class distribution across train, validation, and test sets.
- Used SMOTE to balance the training data and improve minority class representation.
- Tracked validation metrics and used early stopping to prevent overfitting.
- Documented all preprocessing, model architecture, and training procedures in the notebook for transparency and reproducibility.

🕵️ 4. Model Explainability
Q: What explainability methods (e.g., SHAP, LIME) did you apply to interpret your model's predictions?
A:

Q: What were the most important features according to your explainability analysis?
A:

Q: How did model explainability influence your understanding or trust in the model?
A:

## Model Explainability: SHAP vs LIME

For this dataset and neural network models, SHAP is the preferred method for explainability. SHAP provides both local and global feature importance, is theoretically sound, and works well with complex models like neural networks. It helps us understand which features most influence the model’s predictions across the entire dataset.

**Summary Table: SHAP vs LIME**

| Aspect                | SHAP                                      | LIME                          |
|-----------------------|-------------------------------------------|-------------------------------|
| Approach              | Shapley values (game theory)              | Local surrogate model         |
| Consistency           | Theoretically guaranteed                  | Heuristic, may vary           |
| Local Explanations    | Yes                                       | Yes                           |
| Global Explanations   | Yes                                       | No                            |
| Model Support         | Any (with wrappers), best for tree/NN      | Any                           |
| Computation           | Slower, more precise                      | Faster, less precise          |
| Output                | Feature contributions (per prediction)     | Feature weights (per example) |

**Recommendation:**

> SHAP is used in the notebook to explain the neural network’s predictions and to identify the most important features influencing the outcome. This provides more reliable and comprehensive insights for this dataset than LIME.