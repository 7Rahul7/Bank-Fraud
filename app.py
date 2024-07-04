import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Function to load the dataset from an uploaded file
@st.cache_data
def load_uploaded_data(uploaded_file):
    # Determine the file type and read accordingly
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Drop missing values and map the 'isFraud' column
    df.dropna(inplace=True)
    df['isFraud'] = df['isFraud'].map({0: 'No Fraud', 1: 'Fraud'})
    return df

# Cache the model training process
@st.cache_resource
def train_model(df):
    # Preprocess the data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']),
            ('cat', OneHotEncoder(), ['type'])
        ])
    
    X = df.drop('isFraud', axis=1)
    y = df['isFraud'].map({'No Fraud': 0, 'Fraud': 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with preprocessing and model training
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=42))])
    
    clf.fit(X_train, y_train)
    
    return clf, X_train, X_test, y_train, y_test

# Streamlit App
st.title("Fraud Detection App")

# Sidebar: File uploader for CSV and Excel files
st.sidebar.header("Upload CSV or Excel File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

# Load and preprocess the uploaded data
if uploaded_file:
    df = load_uploaded_data(uploaded_file)
else:
    st.warning("Please upload a CSV or Excel file to proceed.")
    st.stop()

# Train and cache the model
clf, X_train, X_test, y_train, y_test = train_model(df)

st.sidebar.header("User Input Features")

# Function to get user input
def get_user_input():
    step = st.sidebar.number_input('Step (time unit)', min_value=0, max_value=744, value=1)
    type = st.sidebar.selectbox('Transaction Type', ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
    amount = st.sidebar.number_input('Transaction Amount', min_value=0.0, value=0.0)
    oldbalanceOrg = st.sidebar.number_input('Original Balance', min_value=0.0, value=0.0)
    newbalanceOrig = st.sidebar.number_input('New Balance After Transaction', min_value=0.0, value=0.0)
    oldbalanceDest = st.sidebar.number_input('Recipient Original Balance', min_value=0.0, value=0.0)
    newbalanceDest = st.sidebar.number_input('Recipient New Balance After Transaction', min_value=0.0, value=0.0)

    user_data = {
        'step': step,
        'type': type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

# Store user input in variable
user_input = get_user_input()

# Display user input
st.subheader('User Input:')
st.write(user_input)

# Preprocess the user input
preprocessed_input = clf.named_steps['preprocessor'].transform(user_input)

# Prediction using the Random Forest model
prediction = clf.named_steps['classifier'].predict(preprocessed_input)
prediction_prob = clf.named_steps['classifier'].predict_proba(preprocessed_input)

# Display Prediction
st.subheader('Prediction:')
st.write('Prediction:', 'Fraud' if prediction[0] else 'No Fraud')
st.write('Prediction Probability:', prediction_prob)

# Displaying model performance metrics only when requested
if st.checkbox('Show model performance metrics'):
    st.subheader('Classification Report:')
    y_pred = clf.predict(X_test)
    st.text(classification_report(y_test, y_pred))
    st.write('ROC-AUC Score:', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

# Visualization of the distribution of the target variable
st.subheader('Fraud Distribution in Dataset')
fig, ax = plt.subplots()
sns.countplot(x='isFraud', data=df)
st.pyplot(fig)

# Visualization of the distribution of the amount for different transaction types
st.subheader('Transaction Amount Distribution by Type')
fig, ax = plt.subplots()
sns.boxplot(x='type', y='amount', data=df)
st.pyplot(fig)

# countplot with bar labels
st.subheader('Count Plot of Transaction Type with Bar Labels')
fig, ax = plt.subplots()
ax = sns.countplot(x='type', data=df, palette='PuBu')
for container in ax.containers:
    ax.bar_label(container)
plt.title('Count plot of transaction type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel('Number of transactions')
st.pyplot(fig)

# heatmap to show the correlation
st.subheader('Correlation Heatmap')
fig, ax = plt.subplots(figsize=(10, 8))
df_numeric = df.copy()
df_numeric['isFraud'] = df_numeric['isFraud'].map({'No Fraud': 0, 'Fraud': 1})
corr = df_numeric.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
plt.title('Heatmap of Feature Correlations')
st.pyplot(fig)
