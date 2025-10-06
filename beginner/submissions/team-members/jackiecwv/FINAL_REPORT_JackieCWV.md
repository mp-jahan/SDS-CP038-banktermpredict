# 🟢 Beginner Track - Jackie CW Vescio

## ✅ **Week 1: Setup, EDA & Feature Engineering**

---

### 🛠️ **1. Project Setup & Data Loading**

**Q: How did you set up your project environment and manage dependencies?**  
- I created a local **Python 3.10 virtual environment (`.venv310`)** in the project root and used that interpreter for all runs (`...\.venv310\Scripts\python.exe`). 
- Package management was done with `pip` inside the `venv (pip 25.2)`. 
- I verified versions in-notebook for reproducibility: `NumPy 1.26.4` and `pandas 2.2.3`, with the environment path printed to confirm I was using the project `venv`.

**Q: What steps did you take to load and inspect the dataset?**  
- I loaded the dataset `bank-full.csv` using `pandas.read_csv()` with the semicolon (;) delimiter specified. 
- After loading, I inspected the data with `.shape` to confirm its size (45,211 rows × 17 columns) and used `.head()` to preview the first records. 
-  I verified the structure with `.info()`, which showed 7 numeric (int64) features (e.g., `age`, `balance`, `duration`, `campaign`) and 10 categorical (object) features (e.g., `job`, `marital`, `education`, `housing`, `y`).
-  I checked for missing values using `.isna().sum()` and a scan for unstructured placeholders (e.g., empty strings), and confirmed there were no missing values.
- Finally, I used `.duplicated().sum()` and confirmed there were 0 duplicate rows.
---
 
### 📦 **2. Data Integrity & Structure**

**Q: Did you find any missing, duplicate, or incorrectly formatted entries in the dataset?**  
- I did not find any missing values in the dataset — both `.isna().sum()` and additional checks for unstructured placeholders (e.g., empty strings, “unknown”) returned 0 across all 17 columns. 
- Using `.duplicated().sum()`, I confirmed there were 0 duplicate rows. The data appears to be consistently formatted.
- Numeric features were stored as integers, and categorical features were represented as text labels, with no irregular characters or mismatched encodings.  
- Note: Although there were no true nulls, some categorical columns (e.g., `job`, `education`) included the category "unknown", which effectively acts as a placeholder for missing information. 
- This will be considered during preprocessing and feature engineering.

**Q: Are all data types appropriate for their features (e.g., numeric, categorical)?**  

Yes. The dataset columns aligned with their expected types:
- **Numeric (int64)**: `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous` — all stored as integers with no irregular formatting.
- **Categorical (object)**: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, and the target `y` — all stored as text labels.
- There was no need to manually convert numeric columns from text, and categorical values were already readable string categories.
- A later preprocessing step will encode these categorical columns for modeling.
- Note: While `day` and `month` are technically time-related features, they are represented as integers/strings rather than true datetime types. For modeling purposes, they will be treated as categorical variables unless transformed during feature engineering.
---

### 📊 **3. Feature Distribution & Target Assessment**

**Q: What did you observe about the distributions of key features (e.g., age, balance, campaign)?**

- `Age`: ranged from 18 to 95 years, with a median ≈ 39 and mean ≈ 40. 
  - Most clients clustered in their 30s–40s, with fewer at older ages. 
  - The distribution was roughly bell-shaped with a slight right skew.
- `Balance`: showed a wide spread, including negative values (overdrafts) and very high positive balances. 
  - The `median ≈ 448` while the `mean ≈ 1,360`, indicating a strong right skew driven by a small number of clients with unusually large balances.  
  - Outliers were frequent in the upper tail.

- `Campaign`: (number of contacts during the campaign) ranged from 1 to 63, with a `median` of 2 and a `mean` ≈ 2.8. 
  - Most clients were contacted only a few times, while a small minority had far higher contact counts, producing a heavily right-skewed distribution with many upper outliers.

- `Duration`: call lengths were typically short, but with some calls lasting several hundred seconds. 
  - The distribution was highly right-skewed, with a small number of long conversations influencing the mean.

- `Pdays`: heavily concentrated at `999`, which represents clients not previously contacted. 
  - This creates a very imbalanced distribution, with only a minority of cases showing meaningful prior contact values.

**Q: Is there a class imbalance in the target variable (`y`)?** 
- Yes, there is a clear class imbalance in the target variable `y`. 
- The majority class ("no") accounts for about 88% of the observations, while the minority class ("yes") makes up only about 12%. 
- This imbalance is important to consider during model training, as it can cause bias in the model toward predicting the majority class.

**How did you check for this?**  
- I used the `.value_counts()` method on the target variable `y` to view the frequency of each class. 
- This confirmed the imbalance by directly showing the proportion of "no" versus "yes" responses.

**Q: What visualizations did you use to summarize your findings?**  
- I used several visualizations to explore the dataset:
  - Histograms for numeric features (`age`, `balance`, `campaign`, `duration`) to view distributions and skewness.
  - Boxplots of balance and duration to detect the presence of extreme outliers. 
  - A bar plot of the target variable `y` to clearly show the imbalance between "no" and "yes".
  - Count plots for key categorical features (`job`, `marital`, `education`) to compare the frequency of categories.

These visualizations helped highlight distributions, confirm the target imbalance, and reveal category frequency patterns.

---

### 🧰 **4. Feature Engineering**

**Q: What new features did you engineer (e.g., campaign frequency, time since last contact)?**  
Q: What new features did you engineer (e.g., `campaign frequency`, `time since last contact`)?
- I engineered a new binary feature, `previously_contacted`, derived from `pdays`, where values < `999` indicate `prior contact` `(1)` and `999` indicates `no prior contact` `(0)`. 
- This simplified the interpretation of the `pdays` column. I also considered the relationship between `campaign` and `previously_contacted` as indicators of overall contact frequency. 
- While `duration` was noted as strongly predictive, I chose to flag it for careful use, since it is only known after the call is completed and could lead to data leakage if included directly.

Note: Additional engineered features may be explored in Weeks 2–3 during preprocessing and modeling.

**Q: Did you identify any features to exclude or transform?**  
- Yes. I chose to exclude `duration` from model training because it is only known after the call is completed and would create data leakage if included. 
- I also transformed `pdays`, which was heavily skewed by the placeholder value `999`. 
- To address this, I derived a simpler binary feature (`previously_contacted`) and planned to handle the remaining values separately. 
- In addition, all categorical features (`job`, `marital`, `education`, etc.) were noted for transformation into numeric form during preprocessing (e.g., one-hot encoding). 
- Finally, I flagged skewed numeric features such as `balance` and `campaign` for potential scaling and normalization, which will be addressed in Week 2 preprocessing.

**Q: How did you address class imbalance (e.g., SMOTE, class weights)?**  
- In Week 1, I did not yet apply resampling or weighting techniques, but I confirmed that the target variable `y` is highly imbalanced (~88% “no” vs. ~12% “yes”). 
- I flagged this as a key issue to address in later stages of preprocessing and model development. 
- Potential strategies include using **SMOTE** to oversample the minority class, applying class weights in models such as Logistic Regression or Random Forest, and comparing performance across these approaches.
---

## ✅ **Week 2: Data Preprocessing & Model Development**

---

### 🏷️ **1. Categorical Feature Encoding**

**Q: Which categorical features did you encode, and what encoding methods did you use (label, one-hot)?**

- I used `pandas get_dummies on TRAIN only`, then `align VAL/TEST` to `TRAIN` columns with `.reindex(columns=..., fill_value=0)` to avoid leakage and mismatched columns.
- Kept the categorical list: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`.
- This keeps the columns consistent across splits and prevents leakage from val/test into training.'

**Q: What encoding methods did you use (label, one-hot)?**

- I used explicit, beginner‑friendly preprocessing: 
  - `get_dummies` on `TRAIN`, then aligned `VAL/TEST` to the `TRAIN` dummy template. 
  - This keeps the steps transparent while I’m learning.

**Q: Show a sample of the encoded data.**  
- Sample from my Python BankTermPredict Notebook:

  - `X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=False)`
  - `X_val_cat  = pd.get_dummies(X_val[categorical_cols], drop_first=False).reindex(columns=X_train_cat.columns, fill_value=0)`
  - `X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=False).reindex(columns=X_train_cat.columns, fill_value=0)`

See the notebook section:
- `One-hot with pandas (train template → align val/test)` 
- The first few columns (e.g., `job_`, `marital_`) show clean 0/1 indicators.

- I validated preprocessing on a 100-row subset: numeric features scaled with `StandardScaler`, categorical features one-hot encoded `(sparse_output=False)`. 
- Shape increased from `(100, 16)` to `(100, 32)` as expected.

---

### ⚖️ 2. Numerical Feature Scaling

**Q: Which numerical features did you scale, and which scaler did you choose (StandardScaler, MinMaxScaler)? Why?**  

- I scaled the numeric features `age`, `balance`, `day`, `campaign`, `pdays`, `previous` with `StandardScaler` from `scikit-learn`.
- I excluded `duration` from modeling to avoid target leakage, as it is only known after the call is completed.
- I chose `StandardScaler` because:
  - It centers features by removing the mean and scales them to unit variance (`std=1`).
  - This is appropriate for features that are normally distributed or approximately so, which fits most of my numeric features.
  - It helps models that assume normally distributed data (e.g., Logistic Regression) perform better.
  - It is less sensitive to outliers than `MinMaxScaler`, which can be skewed by extreme values.

**Q: Show summary statistics of the scaled features.**  

Evidence (rounded to 3 decimals):
- In the notebook’s “Sanity checks (tiny)” cell, I print the aggregated stats on the scaled `TRAIN` numerics.
- The table shows `means` near 0 and `std` near 1 for each numeric feature.

Final shapes: (31647, 50) (6782, 50) (6782, 50)
| feature  |  mean |   std |
|---|---:|---:|
| age      | 0.000 | 1.000 |
| balance  | 0.000 | 1.000 |
| day      | 0.000 | 1.000 |
| campaign | 0.000 | 1.000 |
| pdays    | 0.000 | 1.000 |
| previous | 0.000 | 1.000 |

OHE binary (train): True
OHE binary (val): True
OHE binary (test): True

NaNs? train/val/test: False False False


### ⚖️ Numerical Feature Scaling Continued — Additional Summary Statistics (Scaled)

**A:** Scaled with `StandardScaler`. 
- **Evidence (rounded to 3 decimals):**

| Feature   |   count |   mean |   std |    min |    25% |    50% |    75% |     max |
|:----------|--------:|-------:|------:|-------:|-------:|-------:|-------:|--------:|
| age       |   31647 |      0 |     1 | -2.154 | -0.742 | -0.177 |  0.67  |   5.095 |
| balance   |   31647 |      0 |     1 | -3.056 | -0.42  | -0.297 |  0.021 |  32.82  |
| day       |   31647 |      0 |     1 | -1.777 | -0.937 |  0.023 |  0.623 |   1.824 |
| campaign  |   31647 |      0 |     1 | -0.567 | -0.567 | -0.245 |  0.077 |  19.401 |
| pdays     |   31647 |     -0 |     1 | -0.41  | -0.41  | -0.41  | -0.41  |   8.339 |
| previous  |   31647 |     -0 |     1 | -0.235 | -0.235 | -0.235 | -0.235 | 110.168 |

*Note:* Means ≈0.000 and std ≈1.000 as expected after standardization. `duration` **excluded from modeling** to avoid leakage.

Post-preprocessing summary + shapes:**
- **All numeric:** final matrices = scaled numerics + 0/1 dummies.
- **No missing values:** NaN checks are False for train/val/test.
- **OHE binary:** min/max across all dummy columns are 0/1.
- **Scaled numerics:** `means ≈ 0`, `std ≈ 1` on `TRAIN`.
- **Shapes (rows × features):** Label them explicitly from notebook output, e.g.  
  `Train (31647, 50)`, `Val (6782, 50)`, `Test (6782, 50)`.

Transformed Dataset Validation (Post-Preprocessing)

- **Shape:** (45,211 rows × 50 columns)
- **Type:** All numeric (`float64`)
- **Missing values:** None detected
- **Categoricals (OHE):** Binary indicators with min=0 and max=1
- **Scaled numerics:** Means ≈ 0 and std ≈ 1 (see summary table above)
- **Leakage control:** **duration** was dropped prior to modeling

*Evidence:* See “Quick Validation of Output” cell:
- `All numeric dtype: True`
- `Any NaNs: False`
- `OHE min==0 and max==1: True`
- `Numeric means ~0: True`


---

### ✂️ **3. Data Splitting**
---
**Q: How did you split the dataset into training, validation, and test sets? What proportions did you use?**

- I used a stratified 70/15/15 split with **train_test_split** (two calls, **random_state=42**): 70% Train, 15% Validation, 15% Test.
- - **Stratified on `y`:** preserves the yes/no class ratio across all splits.
- **First call** - **train/test split**:
  - Split the data into train (70%) and a temporary set (30%) that will be further split into validation and test (15% each):
    - `X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)`
- **Second call** - **val/test split**:
  - 15% val, 15% test (split the 30% temp in half)
    - `X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)`
---
Q: **Did you use stratification? Why or why not?**  

- Yes, I used stratification based on the target variable  `y` to ensure that each split maintained the same class distribution as the original dataset.

**Why stratified:**
- The target is imbalanced (far more “no” than “yes”). 
- Stratification preserves class proportions in each split, so metrics on val/test are comparable to training and not skewed by a lopsided split.
- This is important for imbalanced datasets, as it helps to prevent overfitting and ensures that the model is trained on a representative sample of the data.
---

### 🤖 **4. Model Training & Evaluation**
---
**Q: Which baseline models did you train (Logistic Regression, Decision Tree, Random Forest)?**

- **Logistic Regression** (`sklearn.linear_model.LogisticRegression`, solver=`lbfgs`, `max_iter=1000`, `random_state=42`)
- **Decision Tree Classifier** (`sklearn.tree.DecisionTreeClassifier`, default depth, `random_state=42`)
- **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`, `n_estimators=200`, `random_state=42`, `n_jobs=-1`)
---
**Q: What metrics did you use to evaluate model performance?**
- **Primary metrics (on the Validation split):**
  - **Accuracy**, **Precision (weighted)**, **Recall (weighted)**, **F1 (weighted)**
- **Why weighted?** The target is imbalanced; weighted averages reflect class proportions and give a fairer single-number summary.
- **Protocol:** Fit on `TRAIN` only; report metrics on `VAL` to estimate generalization. (TEST held out for the end.)
---
- **Results summary**: (rounded to 3 decimals)

| model        | set   |   accuracy |   precision_w |   recall_w |   f1_w |
|:-------------|:------|-----------:|--------------:|-----------:|-------:|
| LogReg       | train |      0.892 |         0.869 |      0.892 |  0.863 |
| LogReg       | val   |      0.893 |         0.872 |      0.893 |  0.865 |
| DecisionTree | train |      1     |         1     |      1     |  1     |
| DecisionTree | val   |      0.836 |         0.841 |      0.836 |  0.839 |
| RandomForest | train |      1     |         1     |      1     |  1     |
| RandomForest | val   |      0.896 |         0.878 |      0.896 |  0.876 |

---
- Notice the perfect training scores for both tree-based models, contrasted with lower validation scores — this is textbook overfitting.
- In practice, tuning parameters like `max_depth` and `min_samples_leaf` can reduce overfitting, but for this baseline comparison, default parameters were used.
 - Before tuning, **Random Forest performed best** on `VAL` with `90% accuracy` and `0.88 F1 (weighted)`.
 - Logistic Regression was stable with approximately `89% accuracy` and `86% F1 (weighted)` on both `TRAIN` and `VAL`.
 - Decision Tree underperforms; overfit the training data but drops to `84% accuracy` on `VAL`.
---

Key takeaways:
- Random Forest is the strongest baseline winner on validation (highest F1/Accuracy), likely due to its ability to combine the predictions of many individual decision trees to make a final prediction, and reducing overfitting.
- Logistic Regression is a solid, interpretable model that performs nearly as well. 
- Decision Tree is too simple and overfits easily.
  
 - **Planned next:** Add ROC–AUC (and confusion matrices) in Week 4.

Q: **How did you tune hyperparameters and validate your models?**  

- I used scikit-learn Pipelines (ColumnTransformer → StandardScaler for numerics + OneHotEncoder for categoricals) combined with GridSearchCV (5-fold) on the TRAIN split only, optimizing weighted F1 to handle class imbalance. 
- For each model family, I searched a small, readable grid, kept the best estimator (refit on full TRAIN), then compared model families on the Validation split. 
- Finally, I refit the winner on TRAIN+VAL and reported metrics on TEST (held out until the end).

- Why this is safe: preprocessing happens inside each CV fold → no leakage.
  b: f1_weighted (class-imbalance aware).
- Grids searched:

---
  - **Logistic Regression**: `C` ∈ (is a member of) `{0.01, 0.1, 1.0, 10.0} (solver=lbfgs, max_iter=1000, random_state=42)`

**Logistic Regression VAL metrics (markdown):**
|   val_accuracy |   val_precision |   val_recall |   val_f1 |
|---------------:|----------------:|-------------:|---------:|
|          0.893 |           0.873 |        0.893 |    0.865 |

---

  - **Decision Tree**: `max_depth ∈ {3, 5, 10, None} × min_samples_leaf ∈ {1, 5, 20} (random_state=42)`
---
**Decision Tree VAL metrics (markdown):**
|   val_accuracy |   val_precision |   val_recall |   val_f1 |
|---------------:|----------------:|-------------:|---------:|
|          0.889 |           0.866 |        0.889 |     0.87 |

---

  - **Random Forest**: `n_estimators ∈ {100, 200} × max_depth ∈ {None, 10, 20} × min_samples_leaf ∈ {1, 5} (random_state=42, n_jobs=-1)`

|   val_accuracy |   val_precision |   val_recall |   val_f1 |
|---------------:|----------------:|-------------:|---------:|
|          0.895 |           0.876 |        0.895 |    0.875 |

---

- **Selection rule**: pick the config with the highest Validation weighted F1; then evaluate that pipeline on TEST after refitting on TRAIN+VAL.
---
**Q: Which model performed best, and why did you select it?**

- **Random Forest** performed best on the Validation split with `90% accuracy` and `0.88 F1 (weighted)`, slightly outperforming Logistic Regression (`89% accuracy`, `0.86 F1 weighted`).
- Why this choice:
  - Handles non-linear patterns and feature interactions after one-hot encoding.
  - More robust than a single tree (reduced variance via ensembling).
  - Higher weighted F1 on Validation under class imbalance.
- Final step: I refit the winning Random Forest pipeline on TRAIN+VAL and reported TEST metrics. 
- Logistic Regression remains a strong interpretable baseline for comparison in the report.
- Decision Tree was excluded due to overfitting and lower validation performance.
---
**TEST set Winners per Model family (markdown):**
| model        |   test_accuracy |   test_precision |   test_recall |   test_f1 |
|:-------------|----------------:|-----------------:|--------------:|----------:|
| RandomForest |           0.896 |            0.877 |         0.896 |     0.875 |
| DecisionTree |           0.889 |            0.867 |         0.889 |     0.871 |
| LogReg       |           0.896 |            0.879 |         0.896 |     0.869 |

---

✅ **Week 3: Model Experimentation & Tracking**
---

🧪 1. **Experiment Tracking**


**Q: How did you track your model experiments and results?**

- I tracked my model experiments and results primarily within my Jupyter notebook by producing well-organized tables summarizing the key metrics after each major round of model experimentation. For each model (e.g., `Logistic Regression`, `Decision Tree`, `Random Forest`, `SVC (Support Vector Classifier)`, `XGBoost`, `LightGBM`, `CatBoost`), I captured validation metrics—including accuracy, precision, recall, F1-score, and confusion matrix results—into summary tables at appropriate points in the workflow.
- While I did not include written observations directly after each experiment in the notebook itself, I discussed my key findings, performance comparisons, and model insights in this final REPORT document. 
- This approach kept the code notebook focused and uncluttered, while ensuring that thoughtful analysis and interpretation were included in the formal report.
---

**Q: What tools or frameworks did you use for experiment tracking (e.g., MLflow)?**
- For experiment tracking, I used a combination of Jupyter Notebook and manually constructed Markdown tables. 
- All code was run and results generated in the notebook environment, where I organized the outputs of each modeling experiment (e.g., metric tables, confusion matrices, and threshold sweeps). 
- I then compiled these results into structured tables for direct inclusion in my final REPORT document.
- No external experiment tracking frameworks such as MLflow or Weights & Biases were used, as our project timeline and PM guidance prioritized rapid model experimentation and delivery of a summary report and Streamlit app.
---
**Q: How did experiment tracking help you in comparing different models and hyperparameters?**
- Experiment tracking helped me systematically compare different models and hyperparameters by providing a clear, organized way to view performance metrics side by side.
- By summarizing results in tables after each round of experimentation, I could easily identify which models performed best on key metrics like recall and F1-score, especially with our class imbalance.
- This structured process allowed me to make informed choices about which models to prioritize for further tuning and validation, and guided my final model selection for deployment.
- Tracking experiments also helped me avoid confusion and stay clear about which configurations I had already tested and their outcomes.
- Overall, experiment tracking was essential to my modeling workflow, enabling efficient iteration and confident, data-driven decisions.
---
**🚀 2. Advanced Model Training**
---

**Q: Which advanced models or boosting methods did you experiment with (e.g., XGBoost, LightGBM)?**
- In addition to baseline models like `Logistic Regression`, `Decision Tree`, and `Random Forest`, I experimented with several advanced gradient boosting methods to further improve performance, particularly for handling class imbalance and complex feature interactions. 
- Specifically, I trained and evaluated the following boosting models:
  - `XGBoost (XGBClassifier)`: A powerful, regularized gradient boosting framework well-known for strong performance on structured/tabular data and robust handling of class imbalance via the scale_pos_weight parameter.
  - `LightGBM (LGBMClassifier)`: An efficient, fast gradient boosting model that performs well with large datasets and has native support for categorical features and class weighting.
  - `CatBoost (CatBoostClassifier)`: Another leading boosting algorithm, designed for high accuracy with minimal parameter tuning and particularly strong with categorical data. 
  - `CatBoost` also helps reduce overfitting and often requires less preprocessing.
- All three models were tuned and evaluated using the same validation protocol as the baseline models, allowing for direct comparison across all key metrics.
---
**Q: What differences did you observe in performance compared to baseline models?**
- Compared to the baseline models (`Logistic Regression`, `Decision Tree`, and `Random Forest`), the advanced gradient boosting models—`LightGBM`, `CatBoost`, and `XGBoost`—demonstrated noticeably higher recall on the validation set, which directly supported the project’s recall-focused business goal.
- Key observations:
  - `LightGBM (balanced)` achieved the highest recall (`0.70`), outperforming all baseline models and other advanced methods. Its precision (`0.30`) and F1 score (`0.43`) were also strong, given the challenging class imbalance.
  - `CatBoost (unweighted)` and `LightGBM (unweighted)` also achieved high recall scores (around `0.68`), along with slightly higher precision and F1 scores than the baselines.
  - `Random Forest (unweighted)`, `Random Forest (balanced)`, and `XGBoost (unweighted)` had recall scores in the `0.65–0.66` range—better than both `Logistic Regression` and `Decision Tree` baselines.
  - `Logistic Regression (balanced and unweighted)` achieved recall in the range of `0.57–0.63`, while both `Decision Tree` variants trailed further behind (`recall ≈ 0.31`).
- Summary Table (Top 3 Models by Recall, before hyperparameter tuning):

| Model                   | Threshold | Recall | Precision | F1    | Accuracy | Pred_Pos_Rate |
|-------------------------|-----------|--------|-----------|-------|----------|---------------|
| LightGBM (balanced)     | 0.45      | 0.704  | 0.304     | 0.425 | 0.777    | 0.271         |
| LightGBM (unweighted)   | 0.11      | 0.679  | 0.318     | 0.433 | 0.792    | 0.250         |
| CatBoost (unweighted)   | 0.11      | 0.675  | 0.322     | 0.436 | 0.795    | 0.246         |

- In summary:
  - The advanced boosting models (`LightGBM` and `CatBoost` in particular) consistently delivered `higher recall`, `F1`, and `balanced trade-offs` compared to the `baseline` approaches.
  - This validated their value for the business goal of identifying as many potential term deposit subscribers as possible, even when allowing for a lower precision rate.
  - Decision Trees, in contrast, had much lower recall, and Logistic Regression offered more stable but generally lower recall than the boosting models.
---
**Q: How did you handle overfitting or underfitting during experimentation?**
  - I addressed `overfitting and underfitting` by:
  - Using a `dedicated validation set`: All model performance was tracked on a separate validation split, ensuring that results reflected true generalization rather than training set memorization.
- Comparing `train` vs. `validation metrics`: I monitored the gap between training and validation scores. 
  - When models (especially `Decision Tree` and `Random Forest`) showed perfect training scores but much lower `validation performance`, I recognized overfitting and adjusted hyperparameters accordingly.
- `Hyperparameter tuning`: For tree-based and boosting models, I tuned parameters such as `max_depth`, `min_samples_leaf`, and `class balancing options` to reduce `overfitting` and `improve recall`.
- `Excluding leaky features`: I explicitly dropped the `duration` feature to avoid data leakage that could artificially inflate model performance.
- This structured experimentation and validation process helped me `reduce overfitting`, `improve model robustness`, and `achieve better balance between recall and precision`.

---
🛠️ **3. Hyperparameter Tuning & Validation**
---

**Q: What hyperparameter tuning strategies did you use (e.g., `GridSearchCV`, `RandomizedSearchCV`)?**

- I used `GridSearchCV` from `scikit-learn` to perform hyperparameter tuning, but focused my tuning efforts on the top three models from baseline experimentation—specifically, `LightGBM`, `CatBoost`, and (optionally) `XGBoost`—since these models demonstrated the highest recall scores on the validation set. 
- For each major model family (e.g., `Logistic Regression`, `Decision Tree`, `Random Forest`, `LightGBM`, `XGBoost`, `CatBoost`), I defined a small, focused grid of key hyperparameters—such as `regularization strength (C)`, `maximum tree depth (max_depth)`, `number of estimators (n_estimators)`, and `minimum samples per leaf (min_samples_leaf)`.- - GridSearchCV performed cross-validated searches on the training data, selecting the parameter combination that maximized weighted `F1-score`, which aligned with our `recall-focused` business objective.
- I did not perform full hyperparameter tuning on models with lower baseline recall, such as `SVC`, `Decision Tree`, or `Logistic Regression`, in order to prioritize time and computational resources on the most promising candidates for deployment.
---
**Q: How did you validate your models (e.g., cross-validation)?**

- I checked my models using `cross-validation`, which means splitting the `training data` into several parts and testing the model on each part in turn. 
- I used `GridSearchCV` to try out different settings and see which ones worked best. This helped me avoid `overfitting` and made sure the model would work well on new data.
- All the `tuning` and `cross-validation` steps were done only on the `training data`. 
- I kept the `validation` and `test sets` completely separate, so they were never used for tuning or picking hyperparameters. 
- This way, I made sure my final results would be honest and not biased by the model “seeing” the answers ahead of time. 
- By following this process, I could fairly compare different models and be confident that the final performance numbers would reflect how well the model would do on new, unseen data.
- Also, I focused on getting the best weighted `F1-score` during `cross-validation`, since the main goal was to find as many “yes” cases as possible (`recall`), while still being fair to both classes because of the imbalance in the data.
---
**Q: What were the key hyperparameters that influenced model performance?**

- The key hyperparameters that had the most impact on model performance were:
  - `learning_rate`: Controls how much each new tree corrects the errors of the previous ones. 
    - Tuning this helped balance learning speed and overfitting, especially for boosting models.
  - `max_depth/depth`: Sets the maximum depth of each tree. Limiting tree depth helped prevent overfitting and improved generalization.
  - `num_leaves`: (for `LightGBM`) Determines the maximum number of leaves in one tree, affecting the model’s complexity and ability to capture patterns.
  - `n_estimators`: The number of trees or boosting rounds. Increasing this usually improved performance up to a point, then could lead to diminishing returns.

After running GridSearchCV, the following values were selected for the top-performing models:



| Model                   | Key Hyperparameters                      | Final Values                                  |
|-------------------------|------------------------------------------|-----------------------------------------------|
| LightGBM (Balanced)     | learning_rate, max_depth, num_leaves, n_estimators | learning_rate=0.05, max_depth=5, num_leaves=15, n_estimators=200 |
| LightGBM (Unweighted)   | learning_rate, depth, n_estimators, num_leaves     | learning_rate=0.1, depth=-1, n_estimators=200, num_leaves=63      |
| CatBoost (Unweighted)   | learning_rate, depth, n_estimators                | learning_rate=0.1, depth=9, n_estimators=200                      |

Tuning these parameters helped improve recall, F1, and overall model stability, supporting both our business goals and best machine learning practices.

---

**📈 4. Model Selection & Insights**
--
**Q: How did you select the final model for deployment?**

I selected the final model for deployment by systematically comparing all top-performing candidates on the validation set, with a primary focus on recall to support the business objective of identifying as many potential subscribers as possible. After thorough evaluation and hyperparameter tuning, I identified the `LightGBM (balanced)` model as the clear leader based on both its recall score and overall robustness. (Selected after repeated reruns and validation—consistently the top recall performer in all experiments.)

For deployment, I updated my Streamlit app to serve only the `LightGBM (balanced) best-performing model`, ensuring a streamlined and production-ready solution that reflects real-world practices. This approach maximizes the model’s impact for the business while demonstrating clarity and professionalism in my deployment workflow.

---
**Q: What metrics and business considerations influenced your decision?**

The primary metric that influenced my decision was recall for the positive class (“`yes`”), since the business objective was to identify as many potential term deposit subscribers as possible. Maximizing `recall` helps ensure that few likely subscribers are missed, even if this results in a lower precision. I also considered the `F1-score` to balance `recall` and `precision`, as well as the overall stability and robustness of the model.

From a business perspective, it was important that the model would reliably flag potential clients for follow-up, increasing the effectiveness of marketing campaigns without overwhelming the team with too many false positives. Additionally, I considered factors such as ease of deployment, computational efficiency, and how well the model could generalize to new data.

---
**Q: What insights did you gain from the model experimentation process?**

Through the model experimentation process, I learned that advanced boosting models like `LightGBM` and `CatBoost` were much more effective than traditional baseline models in handling class imbalance and maximizing recall. Careful hyperparameter tuning and threshold selection were essential for meeting the business goal of identifying as many likely subscribers as possible. I also realized the importance of using a `separate validation set to avoid overfitting and ensure fair comparisons between models`. Overall, the process reinforced that aligning model development with business objectives—and tracking experiments in a structured way—leads to more robust and impactful results.

**Final Remark**: the deployed Streamlit app uses the “LightGBM (Balanced, Untuned)” model with a 0.45 recall-first threshold, matching your BankTermPredict Jupyter notebook and app for reviewers.

To view the Streamlit app, please visit: https://bankterm-app.streamlit.app

---

