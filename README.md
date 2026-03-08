# Student Performance Predictor - https://student-performance-predictor-ml-end2end.onrender.com

## Project Overview

This repository contains a machine learning application that predicts student performance based on demographic and academic factors. The project demonstrates how data-driven insights can be used to understand and forecast student outcomes, providing a practical example of end-to-end machine learning deployment.

## About the Data

The dataset used in this project is sourced from Kaggle and contains 1,000 records of students, each described by the following features:

- **gender**: Student's gender (male/female)
- **race_ethnicity**: Student's group (A, B, C, D, E)
- **parental_level_of_education**: Highest education level achieved by the student's parents
- **lunch**: Type of lunch (standard or free/reduced)
- **test_preparation_course**: Completion status of a test preparation course (completed/none)
- **math_score**: Score in math exam
- **reading_score**: Score in reading exam
- **writing_score**: Score in writing exam

This data is clean, with no missing or duplicate values, and provides a balanced view of various backgrounds and academic achievements.

## From Data to Prediction

The project begins with exploratory data analysis to understand the relationships between features and student scores. Key insights are drawn about how factors like parental education, test preparation, and lunch type correlate with performance.

After analysis, the data is preprocessed using standard techniques such as encoding categorical variables and scaling numerical features. Multiple regression models are trained and evaluated to identify the best approach for predicting student scores.

The final model is integrated into a web application built with Flask. Users can input student details through a simple web form, and the application returns a predicted performance score based on the trained model.

---

## Components

- **Data Ingestion**: Reads raw data, splits it into training and testing sets, and saves them for further processing.
- **Data Transformation**: Handles preprocessing, including encoding categorical variables and scaling numerical features. The transformation pipeline is saved for use during inference.
- **Model Trainer**: Trains multiple regression models, evaluates their performance, and saves the best-performing model.
- **Utilities**: Includes helper functions for saving/loading models, logging, and exception handling.

## Pipelines

- **Training Pipeline**: Orchestrates the data ingestion, transformation, and model training steps to produce a trained model ready for deployment.
- **Prediction Pipeline**: Loads the trained model and preprocessor, processes user input, and generates predictions for new data.

## API

The project exposes a simple web API using Flask:

- **/** : Home page with a form for user input.
- **/predictdata** : Accepts user input (student details) via POST, processes the data, and returns the predicted performance.

## Deployment Setup

To deploy the application:

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```
   python app.py
   ```
4. Access the web interface at `http://127.0.0.1:5000/`.

For production deployment (e.g., on Heroku or similar platforms), use the provided `Procfile` and ensure all environment variables and dependencies are configured as required.

---

## Purpose

This project serves as a practical demonstration of how machine learning can be applied to real-world educational data, offering insights for educators, policymakers, and anyone interested in predictive analytics.
