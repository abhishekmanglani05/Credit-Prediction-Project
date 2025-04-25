Credit Risk Prediction Using German Credit Dataset – Summary
The objective of this project was to develop a machine learning model capable of predicting the credit risk of loan applicants, classifying them as either good risk or bad risk. Using the German Credit dataset, I followed a structured approach involving data preprocessing, model development, performance evaluation, and insight generation.
________________________________________
1. Data Preprocessing & Exploration
The dataset was first loaded and inspected to understand the distribution of features and the target variable. I removed the unnecessary column 'Unnamed: 0', and handled missing values in the 'Saving accounts' and 'Checking account' columns by filling them with the placeholder 'unknown'. This ensured the dataset was complete and ready for modeling.
To prepare the categorical data for modeling, I used Label Encoding, converting string-based features such as 'Sex', 'Housing', 'Saving accounts', 'Checking account', and 'Purpose' into numerical values. For numerical consistency and to aid in model convergence, I applied StandardScaler to normalize the numerical features: 'Age', 'Credit amount', and 'Duration'.
________________________________________
2. Feature Engineering & Target Definition
Rather than using the original target variable 'Risk', I engineered a new binary target based on financial intuition. If the 'Credit amount' was above the median and 'Duration' was below the median, the applicant was considered high risk (1), otherwise low risk (0). This allowed me to frame the problem in a way that emphasizes financial prudence.
________________________________________
3. Model Development
I experimented with two models:
•	Random Forest Classifier
•	XGBoost Classifier (with GridSearchCV for hyperparameter tuning)
The Random Forest model was trained using the preprocessed dataset and split with an 80-20 train-test ratio. It produced strong results with balanced accuracy (near 100 percent on test data set), good F1-score, and interpretability. To benchmark its performance, I also trained an XGBoost model using GridSearchCV over key hyperparameters like n_estimators, max_depth, and learning_rate. This exhaustive search aimed to maximize the F1-score, ensuring a balance between precision and recall.
Despite XGBoost being a powerful gradient boosting model, the Random Forest model achieved higher accuracy and produced more stable predictions on my dataset, without extensive tuning. Additionally, Random Forests are easier to interpret and deploy, making them a more practical choice for this use case.
________________________________________
4. Evaluation & Insights
Evaluation metrics such as confusion matrix, classification report, and visualizations confirmed that the Random Forest model performed well, particularly in identifying high-risk applicants. From the feature importance output, factors such as Credit Amount, Duration, and Age emerged as key indicators of creditworthiness.
________________________________________
5. Conclusion & Recommendations
The Random Forest model was selected due to its better accuracy, ease of use, and strong generalization. This project not only builds a predictive model but also helps financial institutions understand what contributes most to loan risk. Future improvements could include using original 'Risk' labels, exploring more granular financial behavior, or integrating external credit score data.
This end-to-end solution can help automate credit risk assessment, minimize defaults, and refine lending strategies in a real-world setting.


Links: 
VIDEO LINK – https://drive.google.com/file/d/1yPG6u6wU2WRUySwBE-KG4dUa7Q-XWV5A/view?usp=sharing
LIVE PROJECT LINK –  https://credit-prediction-project.onrender.com/
PROJECT GITHUB LINK – https://github.com/abhishekmanglani05/Credit-Prediction-Project
COLLAB NOTEBOOK LINK / CODE LINK – https://colab.research.google.com/drive/1XJ6HG7LoIocXjvvFOJqQq_0QHBG3jXtt?usp=sharing
https://colab.research.google.com/drive/1XJ6HG7LoIocXjvvFOJqQq_0QHBG3jXtt?usp=sharing
COLLAB PDF - https://drive.google.com/file/d/1FF-Hs9T216bERSgqiA1EOWQkkTRUAYX4/view?usp=sharing
Linkedin Link - https://www.linkedin.com/in/abhishek-manglani-67a300226/
