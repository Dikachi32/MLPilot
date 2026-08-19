MLPilot (ModelForge AI)

    A no-code, browser-based AutoML platform that trains, compares, and exports machine learning models from any CSV dataset in seconds.

MLPilot is a lightweight web application built with Flask and scikit-learn that eliminates the need to write code for standard machine learning workflows. Upload a CSV file, and the system automatically handles data preprocessing, feature engineering, model selection, training, evaluation, and export — all through an intuitive web interface.
Table of Contents

    Features
    Tech Stack
    Project Structure
    Installation
    Usage
        Auto Mode
        Manual Mode
    Supported Models
    Data Preprocessing Pipeline
    API Endpoints
    Problem Type Detection
    Screenshots & UI
    Author
    License

Features
Table
Feature	Description
Zero-Code ML	Train and compare models without writing a single line of code
Auto Mode	Fully automated pipeline — just upload and go
Manual Mode	Full control over features, target, and model selection
Smart Target Detection	Automatically identifies the target column from common naming patterns
Auto Feature Selection	Intelligently filters out ID columns, near-empty columns, and high-cardinality text
Robust Preprocessing	Handles missing values, categorical encoding, datetime parsing, currency stripping, outlier removal, scaling, and PCA
NLP Support	Automatic TF-IDF vectorization for single-column text datasets
Imbalanced Data	Automatic SMOTE application for skewed classification datasets
Ensemble Learning	Voting Classifier / Voting Regressor that combines all trained models
Model Export	Download any trained model (or vectorizer) as a .pkl file
Encoding Detection	Auto-detects CSV file encodings (UTF-8, Latin-1, CP1252, etc.)
Responsive UI	Clean, modern single-page interface with loading states and result rankings
Tech Stack
Table
Layer	Technology
Backend	Python, Flask
Machine Learning	scikit-learn, imbalanced-learn
Data Processing	pandas, numpy, scipy
NLP	scikit-learn TF-IDF Vectorizer
Serialization	joblib
Frontend	HTML5, CSS3, Vanilla JavaScript (Jinja2 templating)
Project Structure
plain

MLPilot/
│
├── app.py                          # Main Flask application — all backend logic
├── .gitignore                      # Git ignore rules
│
├── models/                         # Saved trained models (.pkl files)
│   └── Linear_Regression_xxxx.pkl
│
├── progress/                       # JSON progress artifacts
│   └── xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.json
│
├── static/
│   └── style.css                   # Application styling
│
└── templates/
    └── model.html                  # Single-page UI template (Jinja2)

    Note: The app creates three runtime directories automatically:

        uploads/ — temporarily stores uploaded CSV files
        models/ — stores exported .pkl model files
        vectorizers/ — stores exported TF-IDF vectorizers

Installation
Prerequisites

    Python 3.8+
    pip

Step 1: Clone the repository
bash

git clone https://github.com/Dikachi32/MLPilot.git
cd MLPilot

Step 2: Create a virtual environment (recommended)
bash

python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

Step 3: Install dependencies
bash

pip install flask pandas numpy scikit-learn imbalanced-learn scipy joblib

Step 4: Run the application
bash

python app.py

Step 5: Open in browser
Navigate to:
plain

http://127.0.0.1:5000

Usage
Auto Mode
The fastest way to get results:

    Upload your CSV dataset.
    Select ⚡ Auto Mode.
    Click Upload & Start Training →.
    The system will:
        Detect the target column automatically
        Select relevant features
        Clean and preprocess the data
        Detect the problem type (Classification / Regression / NLP)
        Train all applicable models
        Apply SMOTE if classes are imbalanced
        Build an Ensemble (Voting) model
        Rank all models by performance
    Download the best model (or any model) as a .pkl file.

Manual Mode
For full control over the pipeline:

    Upload your CSV dataset.
    Select 🎛️ Manual Mode.
    Select Feature Columns — check the columns you want as inputs.
    Select Target Column — choose the column you want to predict.
    Choose Auto (train all models) or Manual (pick specific algorithms).
    Click Train Models →.
    Review the ranked results and download your preferred model.

Supported Models
Classification Models
Table
Model	Library
Logistic Regression	sklearn.linear_model.LogisticRegression
Random Forest	sklearn.ensemble.RandomForestClassifier
Gradient Boosting	sklearn.ensemble.GradientBoostingClassifier
AdaBoost	sklearn.ensemble.AdaBoostClassifier
Bagging	sklearn.ensemble.BaggingClassifier
Extra Trees	sklearn.ensemble.ExtraTreesClassifier
Regression Models
Table
Model	Library
Linear Regression	sklearn.linear_model.LinearRegression
Random Forest Regressor	sklearn.ensemble.RandomForestRegressor
Gradient Boosting Regressor	sklearn.ensemble.GradientBoostingRegressor
AdaBoost Regressor	sklearn.ensemble.AdaBoostRegressor
Bagging Regressor	sklearn.ensemble.BaggingRegressor
Extra Trees Regressor	sklearn.ensemble.ExtraTreesRegressor
Ensemble Model

    Voting Classifier (hard voting) — for classification tasks
    Voting Regressor — for regression tasks

Data Preprocessing Pipeline
MLPilot applies a comprehensive preprocessing pipeline automatically:
1. CSV Encoding Detection
Tries 10+ encodings (utf-8, utf-8-sig, latin-1, iso-8859-1, cp1252, cp850, iso-8859-15, mac_roman, utf-16, utf-32) before falling back to replacement mode.
2. Target Column Detection
Scans for common target keywords:
plain

target, label, class, y, outcome, result, prediction, predict,
category, type, status, grade, score, rating, rank, decision,
flag, spam, ham, sentiment, class_label

Falls back to the last column with ≤100 unique values, or the first binary column, or simply the last column.
3. Feature Selection
Automatically excludes columns that are:

    The target column itself

        80% missing values

    Single unique value (constant)
    Unique per row (likely IDs)
    Text columns with >500 unique values (in Auto Mode)

4. Column Type Inference

    Numeric — integers, floats, or text that is >80% convertible to numbers
    Datetime — text that is >80% convertible to dates
    Categorical — everything else

5. Data Cleaning

    Currency stripping — removes $, €, £, ¥, %, and whitespace from numeric-looking strings
    Missing value imputation — median for numeric, "Missing" for categorical
    Categorical encoding:
        Binary (≤2 categories) → Label Encoding
        Low cardinality (≤50 categories) → One-Hot Encoding (drop_first)
        High cardinality (>50 categories) → Frequency Encoding
    Datetime decomposition → year, month, day features
    Target encoding — text targets are label-encoded
    Outlier removal (regression only) — removes rows with Z-score > 3
    Constant column removal — drops zero-variance features
    StandardScaler — z-score normalization
    PCA — reduces to 50 components if features > 50

6. Class Balancing (Classification Only)
If the minority class is <30% of the majority class and has ≥6 samples, SMOTE is applied to generate synthetic samples.
7. Train/Test Split

    Datasets with ≥50 rows → 80/20 split
    Datasets with <50 rows → 70/30 split
    Random state fixed at 42 for reproducibility

API Endpoints
Table
Endpoint	Method	Description
/	GET, POST	Main application page — handles upload, feature selection, and training
/download_model/<model_name>	GET	Downloads a trained model as a .pkl file
/download_vectorizer	GET	Downloads the TF-IDF vectorizer as a .pkl file (NLP mode only)
Problem Type Detection
The app automatically determines the ML task type:
Table
Condition	Detected Type
Only 1 feature column and it's text (object)	NLP Classification
Target has ≤10 unique values	Classification
Target has >10 unique values	Regression
Screenshots & UI
The application features a modern, responsive single-page interface:

    Navbar — branded as "ModelForge AI" with Beta badge
    Hero Section — explains the platform with feature pills
    Upload Card — drag-and-drop style file input
    Mode Selection — visually distinct Auto vs Manual cards
    Feature Table — sortable preview with column type, unique count, and sample data
    Results Dashboard — ranked table with medals (🥇🥈🥉), performance bars, and download buttons
    Info Panels — show auto-selections, cleaning log, SMOTE status, and NLP detection
    Loading Overlay — spinner with "Processing your dataset..." message during training
    Footer — credits, tech stack, and social links

Author
Dikachi Baron
ML Engineer & Developer

    LinkedIn
    GitHub
    X (Twitter)
    Facebook


Future Enhancements

    [ ] Add requirements.txt for easier dependency management
    [ ] Add live prediction endpoint (upload new data + model → get predictions)
    [ ] Support for additional file formats (Excel, JSON, Parquet)
    [ ] Hyperparameter tuning with GridSearchCV / RandomizedSearchCV
    [ ] Cross-validation support
    [ ] Model comparison charts (matplotlib / plotly)
    [ ] Persistent database for training history
    [ ] Docker containerization
    [ ] REST API for programmatic access

    Built with ❤️ using Flask, scikit-learn, pandas, and numpy.