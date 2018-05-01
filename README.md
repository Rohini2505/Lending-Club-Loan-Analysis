# Lending-Club-Loan-Analysis
Explanatory Data Analysis and ML model building using Apache Spark and PySpark

LENDING CLUB DATA ANALYSIS AND DEFAULT LOAN/RATING PREDICTION

OVERVIEW
1.	Introduction
This is a Course project for CISC-5950 Big Data Programming, Fordham University. Under the scope of the course work, we are required to solve an analysis/learning problem using the techniques taught in the course. 
We will use “Lending Club historical dataset” for this project. This is an open source dataset which contains complete loan data for all the loan issued through 2007-2015. The data is available to download on the following site- https://www.kaggle.com/wendykan/lending-club-loan-data
2.	Description of data
Lending Club is an online peer to peer credit marketplace which matches borrowers and investors. For evaluating the credit-worthiness of their borrowers, Lending Club relies on many factors related to borrowers such as credit history, employment, income, ratings etc. Lending club then assign’s rating/sub-rating to their borrowers based on their credit-history. This rating information is then made available to investors who fund the loan requests, investors usage this information to analyze loan request and adjudicate the approved funded amount. In addition to the grade information, Lending Club provides historical loan performance data to investors for more comprehensive analysis. In summary, our dataset contains total of 887,379 records with 75 features in a comma separated file. Each record in the file represent a loan request. This dataset has different types of features such as categorical, continuous. numerical etc. 
3.	Sample data
This dataset contain complete loan data for all loans issued through the 2007-2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. The file containing loan data through the "present" contains complete loan data for all loans issued through the previous completed calendar quarter. Additional features include credit scores, number of finance inquiries, address including zip codes, and state, and collections among others. 
Below is the screenshot of the sample data which includes only few features and first 50 rows, Please check the attached file in blackboard to see all the features in the dataset.
The Sample data file is attached along with the project proposal file in blackboard.
4.	Project Scope
Our project scope is to run the exploratory data analysis using apache spark framework to find the business insights from our loan data, and to build a learning model using data mining techniques / machine leaning algorithms that will use the historic loan data to learn and helps to identify loans/borrowers which are likely to default. 
As per the recent studies, 3-4% of the total loans defaults every year. This is the huge risk for the investors who is funding the loans. Investors require more comprehensive assessment of these borrowers than what is presented by Lending Club in order to make a smart business decision. Data mining techniques and Machine Learning model/analysis could help predicting the loan default likelihood which may allow investors to avoid loan defaults thus limiting the risk of their investments. 
4.1	Data Cleaning and Exploratory Analysis  
The data used for this project is the structured data with few missing/null values. We will use the data cleaning/imputation techniques for the missing values imputation. We may also need to remove some features which have missing values for more than 50% of the data. In addition, we may have to add the “CLASS” in the data manually, since this data doesn’t contains a single column which represents the class. 
After cleaning the data, we will use apache spark RDD/dataframe/sql to run exploratory data analysis to find the business insights from the data. We will extensively use spark’s transformations, actions, aggregation API’s and SQL queries for the data analysis.
4.2	Feature selection
Our dataset contains 75 features for every loan record, using all the features in the model might lead to the “Curse Of Dimensionality”. Therefore, we will use feature selection techniques to discover the features that are most indicative in default loan prediction. We will use spark’s machine leaning library for the built-in feature selection algorithms.
4.3	Prediction Model 
In order to make a smart business decision, investors require a more comprehensive assessment of the loan data provided by Lending Club. As part of this project, we will build a learning model using data mining techniques and Machine Learning algorithms, which helps predicting new borrowers that would likely to default their loans. We will use this dataset as our training set for the model. We will be using apache spark’s machine learning library (MLlib) to build our model.


