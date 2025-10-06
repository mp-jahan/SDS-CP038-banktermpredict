# 🟢 Beginner Track

## ✅ Week 1: Setup, EDA & Feature Engineering

---

### 🛠️ 1. Project Setup & Data Loading

Q: How did you set up your project environment and manage dependencies?  
A: I forked the repo, cloned it locally, and in my folder used python -m venv venv, activated the environment, then pip install -r requirements.txt  

Q: What steps did you take to load and inspect the dataset?  
A: Downloaded a local copy in the data folder, pd.read_csv, .head, .info, .describe, .unique 

---

### 📦 2. Data Integrity & Structure

Q: Did you find any missing, duplicate, or incorrectly formatted entries in the dataset?  
A: All records were non-null (dataset uses 'unknown' values) and rows of feature data had no duplicates.  pdays column has -1 to represent customer was not previously contacted. The yes/no categorical columns need to convert to 1/0, and the three-letter month abbreviations can be encoded.

Q: Are all data types appropriate for their features (e.g., numeric, categorical)?  
A: Education should represent its ordinality.

---

### 📊 3. Feature Distribution & Target Assessment

Q: What did you observe about the distributions of key features (e.g., age, balance, campaign)?  
A: All are highly skewed except age.
   Age is closest to normally distributed, but still has a tail for older (but not impossible) customers.
   Balance has most customers around zero, a few with negative balances, but heavy skewing out to a couple customers near $100,000.
   Campaign has a mean of 2.78 contacts per customer but outliers from around the 10 to 60 range.
   Only 18% of customers were contacted during the previous campaign per the pdays colummn, this can become binary.  

Q: Is there a class imbalance in the target variable (`y`)? How did you check for this?  
A: Yes, the target variable is 'yes' less than 12% of the time.  I used a bar chart to compare the value counts.

Q: What visualizations did you use to summarize your findings?  
A: I used bar charts to compare value counts of y for categorical columns and a correlation heatmap for numerical columns including y as binary.
   Distributions were on histograms and boxplots.

---

### 🧰 4. Feature Engineering

Q: What new features did you engineer (e.g., campaign frequency, time since last contact)?  
A: Split pdays into never_contacted flag and days_since_contacted
   Campaign_per_previous is a ratio of current campaign calls to previous contacts
   An interaction balance * housing (1/0) if higher balance with mortgage might influence product interest

Q: Did you identify any features to exclude or transform?  
A: Duration is out due to known data leakage.  
   Robust scaling is needed for the highly skewed columns.  
   Education becomes ordinal.
   Split days into sine and cosine columns to represent circular nature.
   Age gets binned then one hot encoded along with other categoricals.
   
Q: How did you address class imbalance (e.g., SMOTE, class weights)?  
A: I ran both SMOTE and class weight experiments, and class weights out-performed SMOTE.

---

## ✅ Week 2: Data Preprocessing & Model Development

---

### 🏷️ 1. Categorical Feature Encoding

Q: Which categorical features did you encode, and what encoding methods did you use (label, one-hot)?  
A: I ordinally encoded education.  Once age was binned, that grouping was one-hot encoded along with all other categoricals.  No label encoding.  

Q: Show a sample of the encoded data.  
A:    education_ord	month_nov	poutcome_other	job_blue-collar	age_group_51-65	contact_unknown
45206	   3	         True	      False	         False	            True	            False
45207	   1	         True	      False	         False	            False	            False
45208	   2	         True	      False	         False	            False	            False
45209	   2	         True	      False	         True	            True	            False
45210	   2	         True	      True	         False	            False	            False
 

---

### ⚖️ 2. Numerical Feature Scaling

Q: Which numerical features did you scale, and which scaler did you choose (StandardScaler, MinMaxScaler)? Why?  
A: Age used Standard because it was close to normal.  Day number needs to be cyclical using sine and cosine. 
   The remaining numerical features used RobustScaler because they were highly skewed. 

Q: Show summary statistics of the scaled features.  
A: Scaled X_train header:

            age   balance  campaign  previous  campaign_per_previous  \
24001 -0.460434  0.302304       0.0       0.0                   0.00   
43409 -1.589641  2.709677       1.0       7.0                  -0.75   
20669  0.292371 -0.152627       1.0       0.0                   1.00   
18810  0.668773 -0.332535       4.5       0.0                   4.50   
23130 -0.272233 -0.143041       4.0       0.0                   4.00   

       days_since_contact  balance_x_housing  
24001                 NaN           0.000000  
43409           -0.051282           0.000000  
20669                 NaN           0.472868  
18810                 NaN           0.000000  
23130                 NaN           0.000000  
                age       balance      campaign      previous  \
count  3.616800e+04  36168.000000  36168.000000  36168.000000   
mean  -1.214099e-16      0.674281      0.381967      0.581730   
std    1.000014e+00      2.262521      1.552080      2.408766   
min   -2.154244e+00     -6.245161     -0.500000      0.000000   
25%   -7.427358e-01     -0.277972     -0.500000      0.000000   
50%   -1.781323e-01      0.000000      0.000000      0.000000   
75%    6.687728e-01      0.722028      0.500000      0.000000   
max    5.091500e+00     74.968479     30.500000    275.000000   

       campaign_per_previous  days_since_contact  balance_x_housing  
count           36168.000000         6584.000000       36168.000000  
mean                0.253112            0.154306           1.277328  
std                 1.574310            0.591438           3.759973  
min                -0.996377           -0.994872          -7.862403  
25%                -0.500000           -0.317949           0.000000  
50%                 0.000000            0.000000           0.000000  
75%                 0.500000            0.682051           1.000000  
max                30.500000            3.466667         111.308140   

---

### ✂️ 3. Data Splitting

Q: How did you split the dataset into training, validation, and test sets? What proportions did you use?  
A: Due to the limited data of 45K records, I used k-fold cross-validation and kept the 80/20 train/test split. 

Q: Did you use stratification? Why or why not?  
A: I used stratification to prevent the low representation of the target minority from getting worse, before the training data was balanced with SMOTE. 

---

### 🤖 4. Model Training & Evaluation

Q: Which baseline models did you train (Logistic Regression, Decision Tree, Random Forest)?  
A: LR, RF 

Q: What metrics did you use to evaluate model performance?  
A: Recall, to focus on identifying as many 'y' customers as possible, then accuracy to monitor trade-offs

Q: How did you tune hyperparameters and validate your models?  
A: I narrowed the top three models to LR, CatBoost and LGBM. 

Q: Which model performed best, and why did you select it?  
A: So far, balanced LR recall is the highest at 63%

---

## ✅ Week 3: Model Experimentation & Tracking

---

### 🧪 1. Experiment Tracking

Q: How did you track your model experiments and results?  
A: I defined a list of combinations of preprocessor and model instances, and a results list of mlflow log_param and log_metric.

Q: What tools or frameworks did you use for experiment tracking (e.g., MLflow)?  
A: Only MLflow 

Q: How did experiment tracking help you in comparing different models and hyperparameters?  
A: Organizing the list of performances for side-by-side and sortation. 

---

### 🚀 2. Advanced Model Training

Q: Which advanced models or boosting methods did you experiment with (e.g., XGBoost, LightGBM)?  
A: XGBoost, LightGBM and CatBoost 

Q: What differences did you observe in performance compared to baseline models?  
A: Logistic Regression performed surprisingly close recall at 2nd place to LGBM.  CatBoost 0.616 but has the highest accuracy overall.  Surprisingly, _not_ using one hot encoding on LGBM and CatBoost lowered scores.

Q: How did you handle overfitting or underfitting during experimentation?  
A: Used StratifiedKFold cross-validation and then validated on a separate test set to observe if scores were too spread apart or too low.

---

### 🛠️ 3. Hyperparameter Tuning & Validation

Q: What hyperparameter tuning strategies did you use (e.g., GridSearchCV, RandomizedSearchCV)?  
A: RandomizedSearch for the group, then GridSearch to further refine the best model. 

Q: How did you validate your models (e.g., cross-validation)?  
A: Yes, k-fold validation to find performance averages and maximizing recall during training on multiple folds. 
   After tuning, each model was retrained on the entire training set and tested on the held-out test set. 

Q: What were the key hyperparameters that influenced model performance?  
A: Class weights drove recall improvement, plus increasing depth from 4-5 and learning rate from 0.1 to 0.005.

---

### 📈 4. Model Selection & Insights

Q: How did you select the final model for deployment?  
A: RandomizedSearch had CatBoost with best recall, then GridSearch found the best hyperparameters.

Q: What metrics and business considerations influenced your decision?  
A: Recall in order to find the most customers likely to purchase a term deposit. 

Q: What insights did you gain from the model experimentation process?  
A: That initial performance can change substantially with tuning as CatBoost moved 10 percentage points from 3rd to 1st. 
   And you need strategies to limit and focus experimentation to manage processing time.