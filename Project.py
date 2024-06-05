import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import pickle
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

st.title("Evolution Data Dashboard")

st.header("Questions this project will answer:")
st.write("""
1. How are the different categories distributed in the data?
2. Which numeric variables are most correlated with each other?
3. Which variables are significant predictors for the target variable?
4. How does multicollinearity affect the data and which variables cause it?
5. How accurate are the predictive models based on selected variables?
6. How are the residuals of the model distributed and what does this indicate about the model?
""")

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('?', '')
    return data

data = load_and_preprocess_data('Evolution_DataSets.csv')

st.write("#### First 5 rows of data")
st.write(data.head())

st.write("#### Data information")
buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.write("#### Statistical description of the data")
st.write(data.describe(include='all'))

st.write("#### Checking data for missing values")
st.write(data.isnull().sum())
st.write(data.isna().sum())

def plot_categorical_distribution(data, categorical_cols):
    for col in categorical_cols:
        st.write(f"#### Distribution {col}")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x=col, order=data[col].value_counts().index)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        plt.close(fig)

categorical_cols = [
    'Genus_&_Specie', 'Location', 'Zone', 'Current_Country', 'Habitat', 'Incisor_Size', 
    'Jaw_Shape', 'Torus_Supraorbital', 'Prognathism', 'Foramen_Mágnum_Position', 
    'Canine_Size', 'Canines_Shape', 'Tooth_Enamel', 'Tecno', 'Tecno_type', 'biped', 
    'Arms', 'Foots', 'Diet', 'Sexual_Dimorphism', 'Hip', 'Vertical_Front', 'Anatomy', 
    'Migrated', 'Skeleton'
]
plot_categorical_distribution(data, categorical_cols)

def plot_correlation_matrix(data, numeric_cols):
    corr_matrix = data[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot()
    st.write("""
**Cranial_Capacity and Height:**
- The correlation coefficient between `Cranial_Capacity` and `Height` is 0.85, indicating a strong positive correlation.
- This strong correlation suggests that as `Cranial_Capacity` increases, `Height` tends to increase as well.

Conclusion: `Cranial_Capacity` and `Height` are the most strongly correlated numeric variables, indicating a significant relationship between skull capacity and height in the dataset.
             """)

numeric_cols = ['Cranial_Capacity', 'Height']
plot_correlation_matrix(data, numeric_cols)

def plot_pairplot(data, important_cols, target_col):
    data_sample = data[important_cols].sample(100, random_state=42)
    sns.pairplot(data_sample, hue=target_col)
    st.pyplot()
    st.write("""
**Cranial Capacity vs. biped:**
- Higher `Cranial_Capacity` values are associated with the "modern" category.
- The "yes" category is found in the mid-range of `Cranial_Capacity` values.
- Categories "low probability" and "high probability" have lower `Cranial_Capacity` values.

**Height vs. biped:**
- Similar trends are observed for `Height`.
- Higher `Height` values are associated with the "modern" category.
- The "yes" category is in the mid-range of `Height` values.
- Categories "low probability" and "high probability" have lower `Height` values.

**Time vs. biped:**
- The "modern" category is concentrated in lower `Time` values.
- Categories "low probability" and "high probability" are spread over a wider range of `Time` values but generally have lower `Time`.

Conclusion: `Cranial_Capacity`, `Height`, and `Time` are significantly correlated with `biped`, showing clear distinctions between the target variable categories.
""")

important_cols = ['Cranial_Capacity', 'Height', 'Time', 'Current_Country', 'biped']
plot_pairplot(data, important_cols, 'biped')

def encode_categorical_columns(data):
    label_encoders = {}
    for column in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

data, label_encoders = encode_categorical_columns(data)

def perform_ols(data, target_column, exclude_column):
    features = data.columns.drop([target_column, exclude_column]).tolist()
    formula = f'{target_column} ~ ' + ' + '.join(features)
    model = smf.ols(formula=formula, data=data).fit()
    st.text(model.summary())
    
    st.write('#### Residuals Distribution')
    residuals = model.resid
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    st.pyplot(fig)
    st.write("""
**Shape of Distribution:**
- The histogram of residuals shows a multimodal distribution.
- Several peaks in the graph indicate the presence of multiple clusters in the data.

**Centering of Residuals:**
- Most of the residuals are centered around zero, which is a good sign as it indicates no systematic bias in the model.

**Symmetry of Distribution:**
- The residual distribution is asymmetric with a longer "tail" towards negative values.
- Ideally, residuals should be normally distributed, indicating a good fit of the model. Here, deviations from normality are observed.

**Variance of Residuals:**
- The variance of residuals varies as values increase, indicating potential heteroscedasticity.
- Heteroscedasticity means the model's prediction errors are not consistent across all levels of the independent variables.

Conclusion: The residuals of the model show some deviation from normality and potential heteroscedasticity. While the centering around zero is a good sign, the multimodal and asymmetric distribution of residuals suggests that the model may not fully capture the underlying data patterns and may benefit from further refinement or transformation of variables.
""")
    significant_vars = [var for var in features if model.pvalues[var] < 0.05]
    return significant_vars

def compute_vif(data):
    X = data[['Cranial_Capacity', 'Height', 'Time']]
    X['Intercept'] = 1
    vif = pd.DataFrame()
    vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['Feature'] = X.columns
    return vif

vif = compute_vif(data)
st.write("#### VIF")
st.write(vif)
st.write(""" 
**Cranial Capacity (VIF = 4.0811):**
- The VIF value of 4.0811 indicates moderate multicollinearity.
- This suggests that `Cranial_Capacity` is partially explained by other independent variables in the model.
- While VIF values below 5 are generally acceptable, it's important to monitor this variable for potential multicollinearity effects.

**Height (VIF = 3.542):**
- The VIF value of 3.542 also indicates moderate multicollinearity.
- Similar to `Cranial_Capacity`, `Height` is partially explained by other variables.
- A VIF value below 5 is considered acceptable, but attention should be given to this variable.

**Time (VIF = 1.796):**
- The VIF value of 1.796 indicates low multicollinearity.
- This means that `Time` is minimally correlated with other independent variables, making it a reliable variable for inclusion in the model.
- VIF values below 2 are considered very good and are not a concern.

**Intercept (VIF = 73.8756):**
- The very high VIF value for the intercept is expected.
- The intercept represents the constant in the regression model and always has a high VIF.
- This high VIF for the intercept is not a concern and is a normal part of the model.

Conclusion: The variables `Cranial_Capacity` and `Height` show moderate multicollinearity but are within acceptable limits. The variable `Time` has low multicollinearity, making it a robust variable for the model. The high VIF for the intercept is normal and not a cause for concern.
""")

target_column = 'biped'
exclude_column = 'Genus_&_Specie'
significant_vars = perform_ols(data, target_column, exclude_column)
st.write(f"Selected variables for training: {significant_vars}")

X_selected = data[significant_vars]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

def load_and_evaluate_models(model_filenames, X_test, y_test):
    loaded_models = {name: pickle.load(open(filename, 'rb')) for name, filename in model_filenames.items()}
    for name, model in loaded_models.items():
        st.write(f"### Estimation of the loaded model {name}")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        st.write(f'{name} Accuracy: {accuracy}')
        st.write(f'{name} F1 Score: {f1}')
        st.write(f"Cross-validation of the model {name}")
        cross_val_scores = cross_val_score(model, X_selected, y, cv=5)
        st.write(f'Cross-validation accuracy scores: {cross_val_scores}')
        st.write(f'Average cross-validation accuracy: {cross_val_scores.mean()}')

model_filenames = {
    'RandomForest': 'RandomForest_model.pkl',
    'SVC': 'SVC_model.pkl',
    'GradientBoosting': 'GradientBoosting_model.pkl'
}
load_and_evaluate_models(model_filenames, X_test, y_test)

