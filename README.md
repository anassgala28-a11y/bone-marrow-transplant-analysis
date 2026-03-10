# MEDICAL DECISION SUPPORT APPLICATION
PREDICTING SUCCESS OF PEDIATRIC BONE MARROW TRANSPLANTS WITH 
EXPLAINABLE ML (SHAP)

## Project Overview

This project develops a decision-support application to predict the success of pediatric bone marrow transplants. The system is:

 - Accurate in predicting transplant outcomes

 - Explainable with SHAP visualizations

 - Presented in a user-friendly Streamlit web interface

 - Fully reproducible with GitHub, virtual environments, and CI/CD workflows

The final goal is to assist physicians in clinical decision-making while maintaining transparency.

## Dataset

We used the Bone Marrow Transplant Children Dataset from the UCI repository:

https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+childre

### Details:

 - 187 pediatric patients

 - 37 clinical and demographic features

 - Target variable: survival (1 = survived, 0 = did not survive)

 - Class distribution: ~60% survived, 40% not survived (moderately imbalanced)

## Project Structure
```bone-marrow-transplant-ml/
│
├── notebooks/        # Exploratory Data Analysis
│   └── eda.ipynb
├── src/              # ML pipeline and data processing
│   ├── data_processing.py
│   ├── train_model.py
│   └── evaluate.py
├── app/              # Web interface (Streamlit)
│   └── app.py
├── tests/            # Automated tests
│   └── test_memory.py
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
└── data/            # Dataset folder
