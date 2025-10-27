import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Students' Social Media Addiction Report")

df=pd.read_csv('Students Social Media Addiction.csv')
tab1,tab2, tab3,tab4, tab5=st.tabs(['Overview','Findings', 'About Data' ,'Visualizations' ,'Predictions'])
with tab1:
    st.write(df.head())
    st.header("Basic Statistics")
    st.write(df.describe())
# with an Addicted_Score ranging from 4 to 9 and an average daily usage time of 5.55 hours
with tab2:
    st.write("1. High School Students are the most addicted to social media")
    high_data = {
             'Addicted Score': [4,9],  # Values from 4 to 9 inclusive
             'Average Daily Usage Hours': [3.0,6.50], # Repeat 5.55 for each score
            'Most Used Platforms' : ['Youtube , Tiktok','Instagram , Snapchat']
    }
    highdf = pd.DataFrame(high_data)
    st.write(highdf)
    st.write("2. Mental Health Score Decreases with Increase in Addiction_Score but stays constant when")
    mental_data = {
         'Addicted Score': [2, 5],  # list of values representing the range
         'Average Daily Usage Hours': [1.50, 4.80]  # corresponding values
    }   
    mentaldf = pd.DataFrame(mental_data)
    st.write(mentaldf)
    st.write("3. Academic Performance gets affected if the Average Daily Usage Hours Exceeds 5.2 Hours")
    # selected_data = df[["Affects_Academic_Performance", "Avg_Daily_Usage_Hours"]]
    # selected_data['Affects_Academic_Performance'] = selected_data['Affects_Academic_Performance'].astype('category').cat.codes
    # selected_data = selected_data.set_index('Affects_Academic_Performance')
    # selected_data['Constant_5_2'] = 5.2
    # st.line_chart(selected_data['Avg_Daily_Usage_Hours'])
    st.write("4. While a person is in a relationship (complicated or not) he/she will always have Conflicts over Social Media")
    min_conflicts = df.loc[df.groupby('Relationship_Status')['Conflicts_Over_Social_Media'].idxmin()]
    st.write(min_conflicts[['Relationship_Status', 'Conflicts_Over_Social_Media', 'Mental_Health_Score']])
with tab4:
    st.header("Top 5 Countries with highest Addicted Scores")
    a = df.groupby('Country')['Addicted_Score'].count()
    a = a.sort_values(ascending=False)
    st.bar_chart(a.head(5))
    
    st.header("Top 5 Platforms with Highest Daily Avg Hours")
    b = df.groupby('Most_Used_Platform')['Avg_Daily_Usage_Hours'].count()
    b = b.sort_values(ascending=False)
    st.bar_chart(b.head(5))

    st.header("Top 5 Platforms with Highest Conflicts over Social Media")
    c = df.groupby('Most_Used_Platform')['Conflicts_Over_Social_Media'].count()
    c = c.sort_values(ascending=True)
    st.bar_chart(c.head(5))

with tab5:
    from sklearn.ensemble import RandomForestClassifier
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop(['Addicted_Score', 'Student_ID'], axis=1)
    y = df['Addicted_Score']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)
     
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
     
    st.subheader("Enter details to predict Addicted Score")
    country_options = df['countries'].unique().tolist() if 'countries' in df.columns else []
    platform_options = df['most_used_platform'].unique().tolist() if 'most_used_platform' in df.columns else []
    user_input = {}

    # Single dropdown for 'country'
    user_input['country'] = st.selectbox(
        "Select Country", df['Country'].unique(), key="select_country"
    )
    
    # Single dropdown for 'most_used_platform'
    user_input['Most_Used_Platform'] = st.selectbox(
        "Select Most Used Platform", df['Most_Used_Platform'].unique(), key="select_platform"
    )
    
    # For all other columns, prompt user for input if not already handled
    for col in X_train.columns:
        if col in ['Country', 'Most_Used_Platform']:
            continue
        base_col = col.split('_')[0] if '_' in col else col
        if base_col in categorical_cols:
            options = df[base_col].unique()
            user_input[col] = st.selectbox(f"Select {base_col}", options, key=f"select_{col}")
        else:
            user_input[col] = st.number_input(f"Enter {col}", value=0.0, key=f"number_{col}")
    
    # Construct dataframe as before
    user_input_df = pd.DataFrame([user_input])
    
    # Adjust columns and prediction logic as before
    for col in X_train.columns:
        if col not in user_input_df.columns:
            user_input_df[col] = 0
    user_input_df = user_input_df[X_train.columns]
    
    if st.button("Predict Addicted Score"):
        prediction = rf_classifier.predict(user_input_df)
    
    



















