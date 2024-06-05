import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score, f1_score
import pickle

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('?', '')
    return data

data = load_and_preprocess_data('Evolution_DataSets.csv')

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
    significant_vars = [var for var in features if model.pvalues[var] < 0.05]
    return significant_vars

target_column = 'biped'
exclude_column = 'Genus_&_Specie'
significant_vars = perform_ols(data, target_column, exclude_column)

X_selected = data[significant_vars]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVC': SVC(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    with open(f'{name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')