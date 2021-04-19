
# &#128199; Introduction

This project aims to develop prediction methods for patients' in-hospital mortality using Machine Learning techniques, in addition to analyzing patients' observations to infer the relationship between observations and the in-hospital death (e.g. average Platelets levels and death/survival relationship). 

The data consist of records from 8000 ICU stays. All patients were adults who were admitted for a wide variety of reasons to cardiac, medical, surgical, and trauma ICUs. ICU stays of less than 48 hours have been excluded.
Four thousand records comprise the training data (set A), and the remaining records are the test data (set B). Outcomes (the target column) are provided for the training set and withheld for the test set.

Each observation has an associated time-stamp indicating the elapsed time of the observation since ICU admission in each case, in hours and minutes. Thus, for example, a time stamp of 35:19 means that the associated observation was made 35 hours and 19 minutes after the patient was admitted to the ICU. Moreover, Each record is stored as a comma-separated value (CSV) text file.

The ICU patient's features are illustrated as follows:
- Age (years)

- Blood urea nitrogen (mg/dL) (BUN)

- Serum creatinine (mg/dL)

- Glasgow Coma Score (GCS) 

- Serum glucose (mg/dL) 

- Heart rate (bpm)

- Serum bicarbonate (mmol/L) 

- Hematocrit (%)

- Hehight (m)

- Serum potassium (mEq/L) (K)

- Serum magnesium (mmol/L) (Mg)

- Invasive systolic arterial blood pressure (mmHg) (SysABP) 

- Non-invasive diastolic arterial blood pressure (mmHg) (NIDiasABP)

- Non-invasive mean arterial blood pressure (mmHg) (NIMAP)

- Serum sodium (mEq/L) (Na)

- Platelets (cells/nL)

- Temperature (Celisius)

- Urine output (mL)

- White blood cell count (cells/nL)

- Weight (Kg)

- Invasive diastolic arterial blood pressure (mmHg) (DiasABP)

- Fractional inspired O2 (0-1) (FiO2)

- Invasive mean arterial blood pressure (mmHg) (MAP)

- Partial pressure of arterial CO2 (mmHg) (PaCO2)

- Partial pressure of arterial O2 (mmHg) (PaO2)

- Arterial pH (0-14 Scale)

The outcome (label) that we will investigate in our analysis is the in-hospital mortality: 
- Survived (recovered and ended hospitalization): 0  
- In-hospital death: 1


For more information about the dataset, please refer to the link: https://physionet.org/content/challenge-2012/1.0.0/

# &#128450; Files

All files used are included in the repsotery. The following describes each file brifely:

- NoteBook.ipynb: The main file which contains the data ETL process, exploration and analysis, preprocessing, and modeling.
- set-a: The training data folder. It has 4000 text files (file for each patient) that contain the patients' observations.
- set-b: The testing data folder. It has 4000 text files (file for each patient) that contain the patients' observations.
- cf_matrix.py: Python module used to plot the confution matrex and evaluation metrics.
- Outcomes-a.txt: Text file that contains the record ids and the target labels (Survived:0 / In-hospital_death:1) for the training data (set-a). The file also has other columns but they are execluded.
- Outcomes-b.txt: Text file that contains the record ids and the target labels (Survived:0 / In-hospital_death:1) for the testing data (set-b). The file also has other columns but they are execluded.

# &#128230; Libraries and Packages

The project is done using python3 and the data science packages. Following the libraries used in details:

 - `Pandas`: Fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool. It will help us to view, clean, and apply analysis techniques to the datasets.
- `NumPy`: NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices. It will help us in mathematical calculation and data analysis.
- `Matplotlib` & `Seaborn`: These two libraries are so powerful in visualizing data and showing the relationship between features.
- `Scikit-learn`: Fabulous tools for predictive data analysis. They will lead us to create a fantastic Machine Learning model predicting a continuous-valued attribute as we will see later.
- `Pickle`: The pickle module implements binary protocols for serializing and de-serializing a Python object structure. We used it to save the ML model as a .pkl file.
- `tqdm`: `tqdm` derives from the Arabic word taqaddum (تقدّم) which means "progress". The package is used to print the progress bar for the iterative processes.
- `XGBoost`: `XGBoost` is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.

# &#128202; Work Methodology and Results 

By applying the data analysis techniques and visualization on set-A, we could infer that there are some observations correlated to in-hospital mortality. For instance, the Blood Urea Nitrogen (BUN) level is proportionally correlated with death/survival. Moreover, some features are dependent on each other like the Non-invasive diastolic arterial blood pressure (NIDiasABP) and the Non-invasive mean arterial blood pressure (NIMAP), Age, and Urine output.

After exploration, we worked on building different Machine Learning models using the Sklearn library to predict the in-hospital mortality before the occurrence, based on the patients' observations collected at the ICU. However, the dataset had many obstacles that should be overcome. For example, there were many columns with mostly null values. As a solution, we dropped all columns that have more than 40% nulls. the reason is to maintain the original distribution of each feature and avoid the undesired modification on the dataset information when applying imputation on the mostly-null features.

By dealing with the missing data, the imbalanced classes was another problem. To retain the balance, we created a new distribution for each feature based on the original for the in-hospital death class data only. The new features had the same values' distribution as the original ones. Then we picked a value from each feature randomly and created a record (totally 2892 records created to achieve the balance with the survived class).

Next, we moved to the modeling step using different classification algorithms (Random Forests, Adaboost, k-nearest neighbor, Logistic regression, and XGBoost). The ROC curve (Area under the curve) is used as an evaluation metric, as we are interested in a low rate of false predictions. As a result, the threshold that achieves the highest AUC for each model was calculated. Consequently, we tested the models' performance and found that the XGBoost classifier has the best AUC.

In the end, we created a simple ETL pipeline to extract and transform (process) the set-B data (testing data) to be ready for loading to apply the prediction algorithms. Accordingly, the XGBoost and Adaboost classifiers achieved nearly the same accuracy and AUC. However, Adaboost had better recall where the XGBoost resulted in better precision.

Recommendation for Improvement: The data originally was collected in a time-series manner. By taking the observation's values average we might lose some information that leads to better performance. For more real applications, we recommend applying time-series analysis techniques to get more reliable models and decisions.

# &#8251; References

- https://physionet.org/content/challenge-2012/1.0.0/
- https://xgboost.readthedocs.io/en/latest/
- https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
- https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223
- https://towardsdatascience.com/anova-for-feature-selection-in-machine-learning-d9305e228476



