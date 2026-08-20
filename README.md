MLPilot — Intelligent AutoML & Model Optimization Platform

Data + Pattern Recognition + Decision Making = Intelligent Model Building
MLPilot is a production-quality AutoML SaaS platform that automatically understands your dataset, detects the machine learning problem type, cleans and preprocesses data, selects appropriate algorithms, trains baseline models, evaluates them, optimizes hyperparameters, detects overfitting/underfitting, and recommends the best model — all without writing code.


-- Features --

Intelligent Dataset Understanding
Automatic detection of numeric, categorical, text, datetime, and image columns
Dataset profiling: missing values, duplicates, constant columns, high cardinality
Automatic problem-type detection: Classification, Regression, NLP, Image, Unsupervised
Automatic Algorithm Selection
Classification: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Gradient Boosting, AdaBoost, Extra Trees, Bagging
Regression: Linear, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, KNN, SVR, Gradient Boosting, AdaBoost, Extra Trees, Bagging
Unsupervised: K-Means, Hierarchical Clustering, PCA
NLP: TF-IDF + Logistic Regression, Random Forest, Gradient Boosting, SVM, Naive Bayes
Image: CNN with Keras/TensorFlow
Data Preprocessing Engine
Missing value imputation (median for numeric, mode for categorical)
Encoding: Label, One-Hot, Frequency
Outlier removal (Z-score for regression)
StandardScaler normalization
Intelligent PCA when features > 50
SMOTE for class imbalance (training data only — no leakage)
Text normalization (lowercase, strip whitespace)
Model Optimization
Hyperparameter tuning via RandomizedSearchCV
Baseline vs Optimized comparison with improvement percentage
Cross-validation support
Evaluation Engine
Classification: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, Log Loss
Regression: MSE, MAE, RMSE, R²
Clustering: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
Overfitting/Underfitting Detection with actionable recommendations
SaaS Features
User registration & authentication (Flask-Login)
Subscription tiers: Free, Trial (14 days), Pro Weekly, Pro Monthly
Backend-enforced usage limits (datasets, training runs, optimizations, downloads)
Demo payment system (no real transactions)
Experiment history tracking
Model artifact packaging (ZIP with model + preprocessing + metadata)


-- Architecture --

plain
mlpilot/
├── app.py                      # Flask application entry point
├── config.py                   # Configuration & plan definitions
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── models/                     # Database models
│   ├── user.py                 # User auth, subscription, usage
│   └── experiment.py           # Experiment history
├── routes/                     # Flask blueprints
│   ├── auth.py                 # Register, login, logout, profile
│   ├── main.py                 # Dashboard, builder, train, results
│   ├── subscription.py         # Plans, checkout, demo payment
│   └── api.py                  # JSON API endpoints
├── services/                   # Business logic
│   ├── subscription_service.py # Plan management & entitlements
│   ├── payment_service.py      # Demo payment processing
│   ├── usage_service.py        # Quota enforcement
│   └── history_service.py      # Experiment persistence
├── ml/                         # Machine Learning engine
│   ├── detection.py            # Dataset understanding & problem detection
│   ├── preprocessing.py        # Data cleaning & transformation
│   ├── feature_engineering.py  # Feature selection & engineering
│   ├── evaluation.py           # Metrics & diagnostics
│   ├── optimization.py         # Hyperparameter tuning
│   ├── artifacts.py            # Model packaging
│   ├── pipeline.py             # End-to-end AutoML orchestrator
│   └── models/
│       ├── classification.py   # Classification algorithms
│       ├── regression.py       # Regression algorithms
│       ├── unsupervised.py     # Clustering & PCA
│       ├── nlp.py              # Text processing
│       ├── image_cnn.py        # Image dataset & CNN
│       └── ensemble.py         # Voting ensembles
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── landing.html
│   ├── dashboard.html
│   ├── builder.html
│   ├── results.html
│   ├── history.html
│   ├── auth/
│   │   ├── register.html
│   │   ├── login.html
│   │   └── profile.html
│   └── subscription/
│       ├── plans.html
│       └── checkout.html
├── static/
│   ├── css/style.css           # Professional SaaS styling
│   └── js/app.js               # Client-side interactions
├── tests/                      # Automated test suite
│   ├── test_detection.py
│   ├── test_preprocessing.py
│   ├── test_evaluation.py
│   ├── test_subscription.py
│   └── test_pipeline.py
├── uploads/                    # Uploaded datasets
└── artifacts/                  # Trained models & preprocessing artifacts

-- Installation --

Prerequisites
Python 3.10+
pip
Setup
bash
# Clone or extract the project
cd mlpilot

# Create virtual environment
py -3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set SECRET_KEY to a secure random string

# Run the application
python app.py
The application will be available at http://127.0.0.1:5000


-- Environment Variables --

Table
Variable	Description	Default
SECRET_KEY	Flask secret key for sessions	dev-secret-key-change-me
DATABASE_URL	SQLAlchemy database URI	sqlite:///mlpilot.db
UPLOAD_FOLDER	Dataset upload directory	uploads
ARTIFACT_FOLDER	Model artifact directory	artifacts
MAX_CONTENT_LENGTH	Max upload size in bytes	52428800 (50MB)
FLASK_ENV	Flask environment	development
FLASK_DEBUG	Debug mode	1
DEMO_PAYMENT	Enable demo payment system	True


-- Usage --

1. Register an Account
Navigate to /auth/register
Fill in your details
A free 14-day trial starts automatically
2. Upload a Dataset
Go to the AutoML Builder (/builder)
Upload a CSV file
MLPilot will analyze the dataset and show a profile
3. Configure Training
Select a target column (or leave empty for unsupervised learning)
Optionally enable hyperparameter optimization
Click "Train Models"
4. View Results
MLPilot trains multiple candidate models
Baseline and optimized results are compared
Overfitting/underfitting diagnostics are shown
The best model is recommended
5. Download Models
Download individual .pkl files
Or download the complete package (model + preprocessing + metadata)
6. Manage Subscription
View plans at /subscription/plans
Upgrade via the demo payment flow (no real charges)


-- Running Tests --

bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=ml --cov=routes --cov=services
Demo Payment Flow
Register an account (automatically starts free trial)
Go to Subscription → select Pro Weekly or Pro Monthly
Review the checkout page (clearly marked as DEMO)
Click Complete Demo Payment
Plan activates instantly — no real transaction occurs]

-- Subscription Plans --

Table
Plan	Datasets	Training Runs	Optimizations	Downloads	Max Dataset	CNN	Advanced Optimization
Free	3	10	0	5	10 MB	No	No
Trial	5	20	5	10	25 MB	Yes	Yes
Pro Weekly	50	200	100	200	100 MB	Yes	Yes
Pro Monthly	200	1000	500	1000	500 MB	Yes	Yes

-- Security --

Passwords hashed with Werkzeug
Secure filename validation
File type restrictions (CSV, ZIP)
File size limits enforced
Path traversal protection
Session-based authentication
Backend-enforced subscription entitlements
No hard-coded secrets
Known Limitations
CNN requires TensorFlow installation (included in requirements.txt)
Very large datasets (>500MB) may require streaming or Dask
Real-time training progress streaming not yet implemented
Payment provider integration (Stripe/Paystack) is architected but not wired


-- Future Extensions --

Enterprise tier with team workspaces
Celery/Redis for async training with progress bars
Transfer learning (ResNet, EfficientNet)
Time-series detection and forecasting
Model deployment endpoint (REST API inference)
SHAP/LIME explainability dashboard
Real payment provider integration


-- License --
MIT License — Built with ❤️ by Dikachi Baron


-- Core Philosophy --

Don't ask the user to know machine learning before using the platform.
A beginner should be able to upload a dataset and let MLPilot determine:
What is this data?
What ML problem is this?
What preprocessing is required?
Which algorithms are appropriate?
Which model performs best?
Can the model be improved?
Is it overfitting or underfitting?
Which optimized model should the user use?
How can the user download it?