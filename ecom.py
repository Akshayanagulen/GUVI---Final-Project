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

# Function to load data
def load_data():
    # Replace 'your_dataset.csv' with the path to your CSV file
    data = pd.read_csv('data.csv')
    return data
data=load_data()
df=pd.DataFrame(data)
data_set = df[df.duplicated()]
df1 = df.drop_duplicates().reset_index(drop=True)


# Streamlit App
# Create tabs in the sidebar
selected_tab = st.sidebar.radio("Navigation", ["Home", "EDA Analysis", "Model UI","Linear Regression","Logistic Regression model","Random Forest Classifier"])

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

elif selected_tab == "Linear Regression":
    st.header("Model Building using Machiece Learning : Classification Algorithms")
     # Allow users to select X and y variables
    selected_variable_X = st.selectbox("Select X Variable", df1.columns)
    selected_variable_y = st.selectbox("Select y Variable", df1.columns)
    st.header("1. Linear Regression")

    X = df1[[selected_variable_X]]
    # Target variable
    y = df1[selected_variable_y]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = linear_model.predict(X_test)

    # Display the prediction
    st.subheader('Prediction')
    st.write(f"Predicted has_converted value: {y_pred[0]:.2f}")

    # Evaluate the model on the test set
    y_pred = linear_model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display model evaluation metrics
    st.subheader('Model Evaluation')
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared Score: {r2:.2f}")
    # Scatter plot with regression line
    plt.figure(figsize=(8, 6))

    # Scatter plot
    plt.scatter(X_test, y_test, color='blue', label='Actual values')

    # Regression line
    x_range = np.linspace(X_test.min(), X_test.max(), 100)
    y_range = linear_model.predict(x_range.reshape(-1, 1))
    plt.plot(x_range, y_range, color='red', linewidth=2, label='Regression line')

    plt.title(f"Scatter Plot with Regression Line ({selected_variable_X} vs {selected_variable_y})")
    plt.xlabel(selected_variable_X)
    plt.ylabel(selected_variable_y)
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot(plt)
    
elif selected_tab == "Logistic Regression model":
    
    st.header("Logistic Regression model")
    selected_variable_1 = st.selectbox("Select X Variable", df1.columns)
    selected_variable_2 = st.selectbox("Select y Variable", df1.columns)
    
    X = df1[[selected_variable_1]]
    # Target variable
    y = df1[selected_variable_2]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Logistic Regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred1 = logistic_model.predict(X_test)

    # Evaluate the model
    precision = precision_score(y_test, y_pred1, average='weighted')
    recall = recall_score(y_test, y_pred1, average='weighted')
    accuracy = accuracy_score(y_test, y_pred1)
    f1 = f1_score(y_test, y_pred1, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred1)

    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_test[selected_variable_1], y=y_test, hue=y_test, palette="viridis", s=80)
    plt.title("Scatter Plot with Logistic Regression Line")
    
    # Plot the logistic regression line
    x_values = np.linspace(X_test[selected_variable_1].min(), X_test[selected_variable_1].max(), 100)
    y_values = logistic_model.predict_proba(x_values.reshape(-1, 1))[:, 1]
    plt.plot(x_values, y_values, color='red', linewidth=3, label="Logistic Regression Line")

    plt.xlabel(selected_variable_1)
    plt.ylabel(selected_variable_2)
    plt.legend()
    st.pyplot()

    # Display Evaluation Metrics
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**F1-Score:** {f1:.2f}")

    # Display Confusion Matrix
    st.write("**Confusion Matrix:**")
    st.write(conf_matrix)
    
elif selected_tab == "Random Forest Classifier":
    # Allow users to select X and y variables
    selected_variable_X_rf = st.selectbox("Select X Variable", df1.columns)
    selected_variable_y_rf = st.selectbox("Select y Variable", df1.columns)

    # Features
    X_rf = df1[[selected_variable_X_rf]]
    # Target variable
    y_rf = df1[selected_variable_y_rf]

    # Split the data into training and testing sets
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

    # Create and train the Random Forest Classifier model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_rf, y_train_rf)

    # Make predictions on the test set
    y_pred_rf = rf_model.predict(X_test_rf)

    # Evaluate the model
    accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
    classification_rep_rf = classification_report(y_test_rf, y_pred_rf)
    conf_matrix_rf = confusion_matrix(y_test_rf, y_pred_rf)

    # Display results in Streamlit
    st.header("Random Forest Classifier Model Evaluation")

    # Display Evaluation Metrics
    st.write(f"**Accuracy:** {accuracy_rf:.2f}")

    # Display Classification Report
    st.write("**Classification Report:**")
    st.write(classification_rep_rf)
    # Display Confusion Matrix
    st.write("**Confusion Matrix:**")
    # Display Confusion Matrix
    st.write("**Confusion Matrix:**")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix_rf, annot=True, cmap="Blues", fmt="d", cbar=False, ax=ax)
    st.pyplot(fig)

    # Display Feature Importance
    st.header("Feature Importance")
    feature_importance = pd.Series(rf_model.feature_importances_, index=X_rf.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    # Create and plot the bar chart
    fig, ax = plt.subplots()
    feature_importance.plot(kind='bar')
    ax.set_title('Feature Importance')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    st.pyplot(fig)