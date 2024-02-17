import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# Function to load data
def load_data():
    # Replace 'your_dataset.csv' with the path to your CSV file
    data = pd.read_csv('data.csv')
    return data
data=load_data()
df=pd.DataFrame(data)
data_set = df[df.duplicated()]
df1 = df.drop_duplicates().reset_index(drop=True)
columns_to_remove = ['totals_newVisits','device_isMobile','geoNetwork_latitude', 'geoNetwork_longitude','youtube','sessionQualityDim', 'last_visitId',
       'latest_visit_id', 'visitId_threshold', 'earliest_visit_id',
       'earliest_visit_number', 'latest_visit_number', 'time_earliest_visit',
       'time_latest_visit', 'days_since_last_visit',
       'days_since_first_visit','earliest_source', 'latest_source', 'earliest_medium', 'latest_medium',
       'earliest_keyword', 'latest_keyword', 'earliest_isTrueDirect',
       'latest_isTrueDirect','products_array', 'target_date']

df1 = df1.drop(columns_to_remove, axis=1)


# Streamlit App
# Create tabs in the sidebar
selected_tab = st.sidebar.radio("Navigation", ["Home", "EDA Analysis", "Model UI","Logistic Regression model","Random Forest Classifier","Decision Tree Classifier"])

# Define the content for each tab
if selected_tab == "Home":
    
    st.header("E-commerce Classification | Prediction: Project")
    st.subheader("the dataset given for classification and prediction")
    st.write(df1)

elif selected_tab == "EDA Analysis":
    st.title("Exploratory Data Analysis")
    # Display the first few rows of the dataset
    st.subheader("1. Data Overview")
    st.write(df1.head())
    # Summary statistics
    st.subheader("2. Summary Statistics")
    st.write(df1.describe())
    # Check for missing values
    st.subheader("3. Missing Values")
    st.write(df1.isnull().sum())
    # Correlation 
    st.subheader("4. Correlation Scatter Plot")
    st.write("To determine the correlation between variables, particularly for numeric variables, you can use the Pearson correlation coefficient.")
    # Dropdown list for selecting variables
    selected_variable1 = st.selectbox("Select Variable for Correlation Analysis", df1.columns)
    selected_variable2 = st.selectbox("Select Second Variable for Correlation Analysis", df1.columns)
    st.subheader(f"Correlation Analysis for {selected_variable1} and {selected_variable2}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=selected_variable1, y=selected_variable2, data=df1, ax=ax)
    st.pyplot(fig)
    # Correlation coefficient
    correlation_coefficient = df1[selected_variable1].corr(df1[selected_variable2])
    st.write(f"Correlation Coefficient: {correlation_coefficient:.2f}")

    # Heatmap with selected variables
    st.header(f"Heatmap Correlation Analysis")
    # Correlation Heatmap
    # Assuming 'data_set' is your DataFrame
    numeric_columns = df1.select_dtypes(include='number')
    correlation_matrix = numeric_columns.corr()

    # Create a heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    st.pyplot(plt)

    # Histogram with the selected variable
    st.header(f"Analysis of Devices")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df1['device_deviceCategory'], bins=1000, kde=True, ax=ax)
    plt.xlabel("Device")
    plt.ylabel("Frequency")
    ax.set_ylim(0, 10000)
    st.pyplot(fig)


elif selected_tab == "Model UI":
    st.title("User Interface Model")
    
    selected_count_hit = st.selectbox("Select count_hit value", sorted(df1['count_hit'].unique()))

    # Filter data based on selected count_hit
    selected_data = df1[df1['count_hit'] == selected_count_hit]

    # Display details in the main area
    st.title("Session Details")
    st.write(f"Details for count_hit = {selected_count_hit}")

    # Display count_hit, count_session, and device_deviceCategory
    if not selected_data.empty:
        st.write(selected_data[['count_hit', 'count_session', 'device_deviceCategory']])
        # Button to check if session has converted
        if st.button("Check Conversion"):
            conversion_status = "Converted" if selected_data['has_converted'].values[0] == 1 else "Not Converted"
            st.success(f"Session has {conversion_status}.")
    else:
        st.write("No data available for the selected count_hit value.")

    # Count the number of conversions
    num_conversions = df1['has_converted'].sum()
    no_converted = len(df1) - num_conversions
    st.subheader("Total customer conversion from the given database")
    # Display counts in Streamlit
    st.write(f'Total number of conversions: {num_conversions}')
    st.write(f'Total number of non-conversion: {no_converted}')

    
elif selected_tab == "Logistic Regression model":
    
    st.header("Logistic Regression model")
    st.subheader("Logistic regression model using Count session, Treansaction revenue to check whether converted")
    # Function to plot logistic regression line
    def plot_logistic_regression_line(X, model, ax):
        # Extract coefficients and intercept from the model
        coef = model.coef_
        intercept = model.intercept_
        # Define x values for the line
        x_values = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 100)
        # Calculate y values using logistic regression equation (y = mx + b)
        y_values = (-intercept - coef[0][0] * x_values) / coef[0][1]
        # Plot the line
        ax.plot(x_values, y_values, color='red', linewidth=2)

    # Assuming you have a DataFrame named 'df1' with features and target variable
    # Replace 'transactionRevenue' and 'count_session' with your actual column names
    # Replace 'has_converted' with your target variable column name

    # Features
    X = df1[['transactionRevenue', 'count_session']]

    # Target variable
    y = df1['has_converted']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Create and train the Logistic Regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = logistic_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    # Convert the classification report to a DataFrame
    df_classification_rep = pd.DataFrame(classification_rep).transpose()


    # Display accuracy and classification report
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write("\nConfusion Matrix:")
    st.write(conf_matrix)
    # Display the classification report in Streamlit
    st.write("### Classification Report:")
    st.table(df_classification_rep)

    # Set up the Streamlit app
    st.title('Scatter Plot: Transaction Revenue vs Session Count')

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the scatter plot
    scatter_plot = sns.scatterplot(
        x='transactionRevenue',
        y='count_session',
        hue='has_converted',
        data=df1,
        palette='viridis',  # You can change the color palette if needed
        ax=ax
    )

    # Set plot labels and title
    plt.xlabel('Transaction Revenue')
    plt.ylabel('Session Count')
    plt.title('Scatter Plot: Transaction Revenue vs Session Count')

    # Show the plot in Streamlit
    st.pyplot(fig)
    
elif selected_tab == "Random Forest Classifier":
    # Features
    X = df[['count_session', 'count_hit']]

    # Target variable
    y = df['has_converted']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)

    # Define and train your RandomForestClassifier model
    rf_model = RandomForestClassifier(random_state=1000, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1)
    rf_model.fit(X_train, y_train)
    st.title("Random Forest Classifier")
    st.subheader("Random Forest Classifier Algorithm for analysing between count session and count hit has impact on conversion")
    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    # Convert the classification report to a DataFrame
    df_classification_rep = pd.DataFrame(classification_rep).transpose()

    # Display the evaluation results
    st.write(f"Accuracy: {accuracy:.2f}")
    # Display the classification report in Streamlit
    st.write("### Classification Report:")
    st.table(df_classification_rep)

    # Display confusion matrix (optional)
    st.write("Confusion Matrix:")
    st.write(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

    # Concatenate X_test and y_test
    data = X_test.join(y_test, how='outer')
    data.columns = ['count_session', 'count_hit', 'has_converted']

    # Set up the Streamlit app
    st.title('Scatter Plot: Session Count vs Hit Count')

    # Add a scatter plot using Seaborn
    fig, ax = plt.subplots()
    scatter_plot = sns.scatterplot(
        x='count_session',
        y='count_hit',
        hue='has_converted',
        data=data,
        ax=ax
    )

    # Set plot labels and title
    plt.xlabel('Session Count')
    plt.ylabel('Hit Count')
    plt.title('Scatter Plot: Session Count vs Hit Count')

    # Show the plot
    st.pyplot(fig)

elif selected_tab == "Decision Tree Classifier":
    # Set up the Streamlit app
    st.title('Decision Tree Classifier')
    st.subheader("Decision Tree Classifier for the analysis of Transaction revenue and count hit impact the conversion")

    # Features
    X = df1[['transactionRevenue', 'count_hit']]

    # Target variable
    y = df1['has_converted']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)

    # Create and train the Decision Tree model
    tree_model = DecisionTreeClassifier(random_state=1000)
    tree_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = tree_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    # Convert the classification report to a DataFrame
    df_classification_rep = pd.DataFrame(classification_rep).transpose()


    # Display the evaluation results
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    
    # Display the classification report in a table
    st.write("### Classification Report:")
    st.table(df_classification_rep)

    # Plot the decision tree
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable deprecated warning
    plt.figure(figsize=(15, 10))
    plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=['Not Converted', 'Converted'])
    st.pyplot()


