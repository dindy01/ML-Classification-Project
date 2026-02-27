import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

#file_location
file_path = r'C:\Users\Dindy\Documents\Statistica Numerica\Heart\heart.csv'

#loading dataset into a pd Dataframe
data = pd.read_csv(file_path)

#first rows of the dataset
#print(data.head())

#pre-processing

#checking if there are any missing values
missing_values=data.isnull().sum()
#print("Missing Values in each column: \n", missing_values)

#Categorical Variabals and Encoding
categorical_vars = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']

data_encoded = pd.get_dummies(data,columns = categorical_vars)

#get:dummies returns boolean values so I cast the result to obtain binary form
data_encoded = data_encoded.astype(int)

numerical_cols = data_encoded.select_dtypes(include=[np.number]).columns
#print(data_encoded.head())

#EDA

#Pearson Correlation Matrix
correlation_matrix = data_encoded.corr()

print("Correlation matrix: \n", correlation_matrix)

plt.figure(figsize=(16,14))
sns.heatmap(correlation_matrix, annot = True, fmt=".2f",cmap='coolwarm',linewidths=0.5, annot_kws={"size":10})
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Pearson Correlation Matrix with Annotations")
plt.show()

#donut chart

def create_donut_chart(data,column):
    counts = data[column].value_counts()
    labels = counts.index
    sizes = counts.values
    
    fig,ax = plt.subplots(figsize = (8, 8), subplot_kw=dict(aspect = "equal"))
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle = 140, wedgeprops=dict(width=0.3))
    ax.legend(wedges, labels, title = column, loc = "center left", bbox_to_anchor=(1,0,0.5,1))
    plt.setp(autotexts, size = 10, weight = "bold")
    ax.set_title(f'Donut Chart of {column}')
    plt.show()

# Create donut charts for the categorical variables 
for var in categorical_vars:
    create_donut_chart(data, var)
    
#Histograms

#numerical variables
numerical_vars = ['Age','RestingBP','Cholesterol','MaxHR']

sns.set(style="whitegrid")

#figure for each variable
fig, axes = plt.subplots(2, 2, figsize=(14,10))

#flatten axes for easy interaction
axes = axes.flatten()

#Plot

for i, var in enumerate(numerical_vars):
    sns.histplot(data[var], bins=20, kde=True, ax=axes[i])
    axes[i].set_title(f'Histogram for {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frequency')
    
#Plots adjustments
plt.tight_layout()
plt.show()

#Chest Pain by Sex

plt.figure(figsize=(12,6))
sns.countplot(data=data, x='ChestPainType', hue = 'Sex')
plt.title('Distribution of Chest Pain Type by Sex')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.show()

#Splitting dataset into a training set and a temp set

#Define Features and target

X= data_encoded.drop('HeartDisease', axis = 1)
y = data_encoded['HeartDisease']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

#Splitting temp set into validation set and test set (50% validation, 50% test set)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify = y_temp)

#♣Check size of each test
print(f'Training set size: {X_train.shape[0]} samples')
print(f'Validation set size: {X_val.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')

#Check the distribution of target variable in each set
print("\nTraining set target distribution:")
print(y_train.value_counts(normalize=True))
print("\nValidation set trarget distribution:")
print(y_val.value_counts(normalize=True))
print("\nTest set target distribuition:")
print(y_test.value_counts(normalize=True))

#Linear Regression between HeartDisease and ST_Slope_Up
X = data_encoded[['ST_Slope_Up']]
Y = data_encoded['HeartDisease']

model = LinearRegression()
model.fit(X,y)

#Predictions
y_pred = model.predict(X)

#Stima Coefficienti
coef = model.coef_[0]
intercept = model.intercept_

print(f"Coefficiente di regressione (Slope) per HeartDisease vs ST_Slope_Up: {coef}")
print(f"Interecetta per HeartDisease vs ST_slope_Up: {intercept}")

#R^2
r2 = r2_score(y, y_pred)

#MSE
mse = mean_squared_error(y, y_pred)

print(f"Coefficiente di determinazione (R^2) per HeartDisease vs ST_Slope_Up: {r2}")
print(f'Errore quadratico medio (MSE) per HeartDisease vs ST_Slope_Up: {mse}')

#Scatter plot and regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color = 'blue', label = 'Dati')
plt.plot(X, y_pred, color = 'red', label = 'Retta di regressione')
plt.xlabel('ST_Slope_Up')
plt.ylabel('HeartDisease')
plt.title('Regressione Lineare: HeartDisease vs ST_Slope_Up')
plt.legend()
plt.show()

#Analisi di normalità dei residui
residuals = y - y_pred

#Plot residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel('Residue')
plt.title('Distribuzione dei Residui')
plt.show()

#Normal probability plot (Q-Q plot)
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot dei Residui')
plt.show()

#Shapiro-Wilk test for normality
shapiro_test = stats.shapiro(residuals)
print(f'Shapiro-Wilk Tet per HeartDisease vs ST_Slope_Flat: statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}')

#Linear Regression between HeartDisease and ST_Slope_Flat

X= data_encoded[['ST_Slope_Flat']]
Y= data_encoded['HeartDisease']

#modello
model = LinearRegression()
model.fit(X, y)

#Predizioni
y_pred = model.predict(X)

#Coeff
coef = model.coef_[0]
intercept = model.intercept_

print(f'Coefficiente di regressione (Slop) per HeartDisease vs ST_Slope_Flat: {coef}')
print(f'Intercetta per HeartDisease vs ST_Slope_Flat: {intercept}')
    
#R^2
r2 = r2_score(y, y_pred)

#MSE 
mse = mean_squared_error(y, y_pred)

print(f"Coefficiente di determinazione (R^2) per HeartDisease vs ST_Slope_Flat: {r2}")
print(f"Errore quadratico medio (MSE per HeartDisease vs ST_Slope_Flat: {mse}")

#Plot e linea di regressione
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label= 'Dati')
plt.plot(X, y_pred, color = 'red', label="Retta di regressione")
plt.xlabel('ST_Slope_Flat')
plt.title('Regressione Lineare: HeartDisease vs ST_Slope_Flat')
plt.legend()
plt.show()

#Normalità residue
residuals = y - y_pred

#Plot residui
plt.figure(figsize=(10,6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel('Residui')
plt.title('Distribuzione dei Residui')
plt.show()

#Q-Q plot
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot dei Residui')
plt.show()

#Test di Shapiro-Wilk per la normalità
shapiro_test = stats.shapiro(residuals)
print(f'Shapiro-Wilk Test per HeartDisease vs ST_Slope_Flat: statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}')

#Classification Models

#ME and MR
def calculate_me_mr(y_true, y_pred):
    me = np.sum(y_true != y_pred)
    mr = me / len(y_true)
    return me, mr

#Hyperparameter Tuning for SVM with GridSearchCV

param_grid = {
    'C': [0.1,1],
    'gamma': [0.1, 0.01],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose = 2, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

#Best parameters from grid search
print('Best parameters found: ', grid.best_params_)

#Predict on the validation set using the best model
y_val_pred_best_svm = grid.best_estimator_.predict(X_val)
me_best_svm, mr_best_svm = calculate_me_mr(y_val, y_val_pred_best_svm)

print("Best SVM Model Performance:")
print(f"Misclassification Error (ME): {me_best_svm}")
print(f"Misclassification Rate (MR): {mr_best_svm:.4f}")
print(classification_report(y_val, y_val_pred_best_svm))


#Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_val_pred_log_reg = log_reg.predict(X_val)
me_log_reg, mr_log_reg = calculate_me_mr(y_val, y_val_pred_log_reg)
print("Logistic Regression Performance:")
print(f"Misclassification Error(ME): {me_log_reg}")
print(f"Miscalssification Rate(MR): {mr_log_reg: .4f}")
print(classification_report(y_val, y_val_pred_log_reg))

# Studio Statistico
def run_model_iterations(k=25):
    misclassification_rates_svm = []
    misclassification_rates_log_reg = []
    
    for i in range(k):
        # Dataset in a pd df
        data = pd.read_csv(file_path)
        
        #Dataset a 10% della size
        data_reduced = data.sample(frac=0.1, random_state=42+i)
        
        #Pre-processing
        categorical_vars = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        data_encoded = pd.getdummies(data_reduced, columns=categorical_vars)
        data_encoded = data_encoded.astype(int)
        
        #Splitting dataset in training set and temp set
        X = data_encoded.drop('HeartDisease', axis=1)
        y = data_encoded['HeartDisease']
        X_train,X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        #Hyperparameter Tuning SVM with GridSearch
        param_grid = {
            'C': [0.1,1],
            'gamma': [0.1, 0.01],
            'kernel': ['linear', 'rbf']
        }
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose = 2, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        #Predict on the test set using the best SVM model
        y_test_pred_best_svm = grid.best_estimator_.predict(X_test)
        me_best_svm_test, mr_best_svm_test = calculate_me_mr(y_test, y_test_pred_best_svm)
        misclassification_rates_svm.append(mr_best_svm_test)
        
        #Logistic Regression
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_reg.fit(X_train, y_train)
        y_test_pred_log_reg = log_reg.predict(X_test)
        me_log_reg_test, mr_log_reg_test = calculate_me_mr(y_test, y_test_pred_log_reg)
        misclassification_rates_log_reg.append(mr_log_reg_test)


        return np.array(misclassification_rates_svm), np.array(misclassification_rates_log_reg)
    
#Iterazioni modello
k = 25
mr_svm, mr_log_reg = run_model_iterations()

#Statistiche descrittive
print("Descriptive Statistics for SVM misclassification Rates:")
print(f"Mean: {np.mean(mr_svm):.4f}")
print(f"Standard Deviation: {np.std(mr_svm):.4f}")
print(f"Min: {np.min(mr_svm):.4f}")
print(f"Max: {np.max(mr_svm):.4f}")

print("\nDescriptive Statistics for Logistic Regression Misclassification Rates: ")
print(f"Mean: {np.mean(mr_log_reg):.4f}")
print(f"Standard Deviation: {np.std(mr_log_reg):.4f}")
print(f"Min: {np.min(mr_log_reg):.4f}")
print(f"Max: {np.max(mr_log_reg):.4f}")

#Histograms
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.hist(mr_svm, bins=5, edgecolor = 'k')
plt.title('SVM Misclassification Rates')
plt.xlabel('Misclassification Rate')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
plt.hist(mr_log_reg, bins = 5, edgecolor='k')
plt.title('Logistic Regresssion Misclassification Rates')
plt.xlabel('Misclassification Rate')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#Boxplots
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
sns.boxplot(y=mr_svm)
plt.title('SVM Misclassification Rates')

plt.subplot(1, 2, 2)
sns.boxplot(y=mr_log_reg)
plt.title('Logistic Regrression Misclassification Rates')

plt.tight_layout()
plt.show()

#Statistica Inferenziale e intervalli di confidenza
alpha = 0.05
mean_mr_svm = np.mean(mr_svm)
mean_mr_log_reg = np.mean(mr_log_reg)
ci_svm = stats.t.interval(1-alpha, len(mr_svm) - 1, loc=mean_mr_svm, scale=stats.sem(mr_svm))
ci_log_reg = stats.t.interval(1 - alpha, len(mr_log_reg) - 1, loc= mean_mr_log_reg, scale=stats.sem(mr_log_reg))

print(f'95% Confidence Interval for SVM Misclassification Rate: {ci_svm}')
print(f'95% Confidence Interval for Logistic Regression Misclassification Rate: {ci_log_reg}')
