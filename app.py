from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
import io
import base64

import matplotlib.pyplot as plt
from flask_cors import CORS
import seaborn as sns
import os
import logging  # Import the logging module

# Ensure matplotlib uses a non-GUI backend for server environments
import matplotlib
matplotlib.use('Agg')

# --- Configuration ---
MAX_WORKERS = os.cpu_count() * 2  # Adjust based on your needs/resource limits
UPLOAD_FOLDER = 'uploads'
MODEL_REGRESSION_FILE = 'regression_model.pkl'
MODEL_CLASSIFICATION_FILE = 'classification_model.pkl'
MODEL_CLUSTERING_FILE = 'clustering_model.pkl'

# --- Logging ---
logging.basicConfig(level=logging.INFO,  # Adjust level as needed
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Application Setup ---
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Thread Pool ---
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# --- Global Data (Carefully Managed) ---
#  Consider using a database or other persistent storage for production.
#  Using in-memory storage like this has limitations (e.g., data loss on restart).
class DataStore:
    def __init__(self):
        self.df = None
        self.X_reg_train_scaled = None
        self.X_reg_test_scaled = None
        self.y_reg_train = None
        self.y_reg_test = None
        self.X_cls_train_scaled = None
        self.X_cls_test_scaled = None
        self.y_cls_train = None
        self.y_cls_test = None
        self.label_encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.lock = threading.Lock()  # For thread-safe access

    def set_data(self, df, X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test,
                 X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test,
                 label_encoders, numerical_cols, categorical_cols):
        with self.lock:
            self.df = df
            self.X_reg_train_scaled = X_reg_train_scaled
            self.X_reg_test_scaled = X_reg_test_scaled
            self.y_reg_train = y_reg_train
            self.y_reg_test = y_reg_test
            self.X_cls_train_scaled = X_cls_train_scaled
            self.X_cls_test_scaled = X_cls_test_scaled
            self.y_cls_train = y_cls_train
            self.y_cls_test = y_cls_test
            self.label_encoders = label_encoders
            self.numerical_cols = numerical_cols
            self.categorical_cols = categorical_cols

    def get_data(self):
        with self.lock:  # Ensure thread-safe access
            return (self.df, self.X_reg_train_scaled, self.X_reg_test_scaled, self.y_reg_train, self.y_reg_test,
                    self.X_cls_train_scaled, self.X_cls_test_scaled, self.y_cls_train, self.y_cls_test,
                    self.label_encoders, self.numerical_cols, self.categorical_cols)

data_store = DataStore()


# --- Utility Functions ---
def generate_confusion_matrix_image(y_true, y_pred, labels, model_name):
    """Generates and encodes a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64


def generate_cluster_distribution_plot(df):
    """Generates and encodes a cluster distribution plot."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Cluster', data=df)
    plt.title('Distribution of Clusters')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64


def generate_box_plot_image(df, col):
    """Generates and encodes a box plot image."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Cluster', y=col, data=df)
    plt.title(f'{col} distribution across clusters')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64


# --- Data Processing Function ---
def process_data(file):
    """Processes the uploaded data and prepares it for modeling."""
    try:
        df = pd.read_csv(file)

        # Check for NaN values *before* any processing
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            raise ValueError(f'NaN values found in columns: {", ".join(nan_cols)}')

        # Drop columns with too many missing values
        df = df.dropna(thresh=len(df) * 0.6, axis=1)

        # Fill missing values
        for col in df.select_dtypes(include=['number']).columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Encode categorical columns
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        # Select features and target dynamically
        X_reg, y_reg = None, None
        X_cls, y_cls = None, None
        if len(numerical_cols) >= 2:
            X_reg = df[numerical_cols[:-1]]  # All but last numeric column as features
            y_reg = df[numerical_cols[-1]]   # Last numeric column as target
        if len(numerical_cols) >= 3:
            X_cls = df[numerical_cols[:-1]]  # All but last numeric column as features
            y_cls = df[numerical_cols[-1]]   # Last numeric column as target

        X_cluster = df[numerical_cols] if len(numerical_cols) >= 2 else None

        # Train-test split
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = (train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
                                                            if X_reg is not None else (None, None, None, None))

        X_cls_train, X_cls_test, y_cls_train, y_cls_test = (train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
                                                            if X_cls is not None else (None, None, None, None))

        # Standardization
        scaler = StandardScaler()
        X_reg_train_scaled, X_reg_test_scaled = (scaler.fit_transform(X_reg_train), scaler.transform(X_reg_test)
                                                    if X_reg is not None else (None, None))
        X_cls_train_scaled, X_cls_test_scaled = (scaler.fit_transform(X_cls_train), scaler.transform(X_cls_test)
                                                    if X_cls is not None else (None, None))

        data_store.set_data(df, X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test,
                             X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test,
                             label_encoders, numerical_cols, categorical_cols)

        return True, None  # Success

    except Exception as e:
        logging.exception("Error during data processing:")
        return False, str(e)  # Failure


# --- API Endpoints ---
import threading

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and data processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_content = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        # Use a thread to process the data without blocking the main thread
        future = executor.submit(process_data, file_content)
        success, error_message = future.result()  # Wait for the result

        if success:
            return jsonify({'message': 'File uploaded and processed successfully'}), 200
        else:
            return jsonify({'error': error_message}), 400

    except Exception as e:
        logging.exception("Error during file upload:")
        return jsonify({'error': str(e)}), 500


@app.route('/regression', methods=['POST'])
def run_regression():
    """Runs the selected regression model."""
    model_name = request.json.get('model')
    models_reg = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVM": SVR(),
        "Gaussian Process": GaussianProcessRegressor(kernel=RBF(), random_state=42)
    }

    if model_name not in models_reg:
        return jsonify({'error': 'Invalid regression model'}), 400

    (df, X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test,
     X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test,
     label_encoders, numerical_cols, categorical_cols) = data_store.get_data()

    if X_reg_train_scaled is None or X_reg_test_scaled is None or y_reg_train is None or y_reg_test is None:
        return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

    model = models_reg[model_name]
    try:
        model.fit(X_reg_train_scaled, y_reg_train)
        y_pred = model.predict(X_reg_test_scaled)

        results = {
            "MAE": mean_absolute_error(y_reg_test, y_pred),
            "MSE": mean_squared_error(y_reg_test, y_pred),
            "R2 Score": r2_score(y_reg_test, y_pred)
        }
        return jsonify(results), 200

    except Exception as e:
        logging.exception(f"Error running {model_name}:")
        return jsonify({'error': f"Error running {model_name}: {str(e)}"}), 500


@app.route('/classification', methods=['POST'])
def run_classification():
    """Runs the selected classification model."""
    model_name = request.json.get('model')
    models_cls = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True),  # Enable probability for ROC AUC
        "Gaussian Naive Bayes": GaussianNB(),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "Gaussian Process": GaussianProcessClassifier(kernel=RBF(), random_state=42),
        "MLP Classifier": MLPClassifier(random_state=42) # Neural Network
    }

    if model_name not in models_cls:
        return jsonify({'error': 'Invalid classification model'}), 400

    (df, X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test,
     X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test,
     label_encoders, numerical_cols, categorical_cols) = data_store.get_data()

    if X_cls_train_scaled is None or X_cls_test_scaled is None or y_cls_train is None or y_cls_test is None:
        return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

    model = models_cls[model_name]
    try:
        model.fit(X_cls_train_scaled, y_cls_train)
        y_pred = model.predict(X_cls_test_scaled)
        y_pred_proba = model.predict_proba(X_cls_test_scaled)  # Get probabilities for ROC AUC

        metrics = {
            "Accuracy": accuracy_score(y_cls_test, y_pred),
            "Precision": precision_score(y_cls_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_cls_test, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_cls_test, y_pred, average='weighted', zero_division=0)
        }

        # ROC AUC (only if binary classification)
        roc_auc = "N/A"
        if len(np.unique(y_cls_train)) == 2:
            try:
                roc_auc = roc_auc_score(y_cls_test, y_pred_proba[:, 1])
            except Exception as e:
                logging.exception("Error calculating ROC AUC:")
        metrics["ROC AUC"] = roc_auc

        # Generate and encode confusion matrix plot
        labels = label_encoders[categorical_cols[-1]].classes_ if categorical_cols else None
        confusion_matrix_image = generate_confusion_matrix_image(y_cls_test, y_pred, labels, model_name)

        return jsonify({'metrics': metrics, 'confusion_matrix': confusion_matrix_image}), 200

    except Exception as e:
        logging.exception(f"Error running {model_name}:")
        return jsonify({'error': f"Error running {model_name}: {str(e)}"}), 500


@app.route('/clustering', methods=['GET'])
def run_clustering():
    """Runs the clustering algorithm and returns analysis."""
    (df, X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test,
     X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test,
     label_encoders, numerical_cols, categorical_cols) = data_store.get_data()

    if df is None:
        return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Setting n_init explicitly avoids a warning
        df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

        # Analyze cluster characteristics
        cluster_analysis = df.groupby('Cluster')[numerical_cols].mean().to_dict('index')

        # Generate cluster distribution plot
        cluster_distribution_plot = generate_cluster_distribution_plot(df)

        # Generate box plots for each numerical column
        box_plot_images = {}
        for col in numerical_cols:
            box_plot_images[col] = generate_box_plot_image(df, col)

        # Most distinctive features (as before)
        def get_most_distinctive_features(cluster_id, top_n=3):
            cluster_data = df[df['Cluster'] == cluster_id][numerical_cols]
            overall_mean = df[numerical_cols].mean()
            cluster_mean = cluster_data.mean()
            feature_importance = abs(cluster_mean - overall_mean)
            most_important = feature_importance.nlargest(top_n)
            return most_important.index.tolist()

        distinctive_features = {}
        for i in range(3):  # Assuming 3 clusters
            distinctive_features[i] = get_most_distinctive_features(i)


        return jsonify({
            'cluster_analysis': cluster_analysis,
            'cluster_distribution_plot': cluster_distribution_plot,
            'box_plot_images': box_plot_images,
            'distinctive_features': distinctive_features
        }), 200

    except Exception as e:
        logging.exception("Error during clustering:")
        return jsonify({'error': str(e)}), 500

import threading
if __name__ == '__main__':
    # Deploy on Render:
    #  - Set environment variable PORT=10000 (or whatever port you choose)
    #  - Bind to 0.0.0.0 to listen on all public IPs

    port = int(os.environ.get('PORT', 5000)) # Default to 5000 if PORT not set
    app.run(host='0.0.0.0', port=port, debug=False) # Remove debug=True for production





# from flask import Flask, request, jsonify, session, make_response
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.svm import SVR, SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
# from sklearn.cluster import KMeans
# import io
# import base64
# import matplotlib.pyplot as plt
# from flask_cors import CORS
# import seaborn as sns
# import os
# import logging
# import secrets  # For generating a secure secret key


# # Ensure matplotlib uses a non-GUI backend
# import matplotlib
# matplotlib.use('Agg')

# app = Flask(__name__)
# CORS(app)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Configure session
# secret_key = secrets.token_hex(16)  # Generate a random 32-character hex string
# app.secret_key = secret_key  # Store in environment variable for production

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# def process_data(file):
#     """Processes the uploaded data and prepares it for modeling."""
#     try:
#         df = pd.read_csv(file)
#         logger.info("Dataframe loaded successfully")

#         # Check for NaN values
#         if df.isnull().any().any():
#             nan_cols = df.columns[df.isnull().any()].tolist()
#             raise ValueError(f'NaN values found in columns: {", ".join(nan_cols)}')

#         # Drop columns with too many missing values
#         df = df.dropna(thresh=len(df) * 0.6, axis=1)

#         # Fill missing values
#         for col in df.select_dtypes(include=['number']).columns:
#             df[col].fillna(df[col].median(), inplace=True)
#         for col in df.select_dtypes(include=['object']).columns:
#             df[col].fillna(df[col].mode()[0], inplace=True)

#         # Identify numerical and categorical columns
#         numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
#         categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

#         # Encode categorical columns
#         label_encoders = {}
#         for col in categorical_cols:
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col].astype(str))
#             label_encoders[col] = le

#         # Select features and target dynamically
#         X_reg, y_reg = None, None
#         X_cls, y_cls = None, None
#         X_cluster = None

#         if len(numerical_cols) >= 2:
#             X_reg = df[numerical_cols[:-1]]  # All but last numeric column as features
#             y_reg = df[numerical_cols[-1]]   # Last numeric column as target

#         if len(numerical_cols) >= 3:
#             X_cls = df[numerical_cols[:-1]]  # All but last numeric column as features
#             y_cls = df[numerical_cols[-1]]   # Last numeric column as target

#         X_cluster = df[numerical_cols] if len(numerical_cols) >= 2 else None

#         X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test = None, None, None, None
#         X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test = None, None, None, None

#         # Train-test split
#         if X_reg is not None:
#             X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

#             # Standardization
#             scaler = StandardScaler()
#             X_reg_train_scaled = scaler.fit_transform(X_reg_train)
#             X_reg_test_scaled = scaler.transform(X_reg_test)

#         if X_cls is not None:
#             X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

#             # Standardization
#             scaler = StandardScaler()
#             X_cls_train_scaled = scaler.fit_transform(X_cls_train)
#             X_cls_test_scaled = scaler.transform(X_cls_test)

#         # Store processed data in the request context
#         data = {
#             'df': df,
#             'X_reg_train_scaled': X_reg_train_scaled,
#             'X_reg_test_scaled': X_reg_test_scaled,
#             'y_reg_train': y_reg_train,
#             'y_reg_test': y_reg_test,
#             'X_cls_train_scaled': X_cls_train_scaled,
#             'X_cls_test_scaled': X_cls_test_scaled,
#             'y_cls_train': y_cls_train,
#             'y_cls_test': y_cls_test,
#             'label_encoders': label_encoders,
#             'numerical_cols': numerical_cols,
#             'categorical_cols': categorical_cols
#         }
#         return data  # Return the processed data as a dictionary

#     except Exception as e:
#         logger.error(f"Error processing data: {e}")
#         raise


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     """Handles file upload and data processing."""
#     logger.info("Received request to /upload")
#     logger.info(f"Request method: {request.method}")
#     logger.info(f"Request headers: {request.headers}")
#     try:
#         if 'file' not in request.files:
#             logger.warning("No file part in the request")
#             return jsonify({'error': 'No file part'}), 400

#         file = request.files['file']
#         if file.filename == '':
#             logger.warning("No selected file")
#             return jsonify({'error': 'No selected file'}), 400

#         data = process_data(io.StringIO(file.stream.read().decode("UTF8", errors='ignore'), newline=None))  # Passes the file to process_data function
#         session['data'] = data  # Storing processed data in session
#         response = jsonify({'message': 'File uploaded and processed successfully'})
#         response.headers['Content-Type'] = 'application/json'
#         return response, 200

#     except ValueError as e:
#         logger.error(f"Validation error: {e}")
#         return jsonify({'error': str(e)}), 400
#     except Exception as e:
#         logger.exception("An unexpected error occurred during file upload")
#         return jsonify({'error': str(e)}), 500


# def get_data():
#     """Helper function to retrieve processed data from the session."""
#     try:
#         data = session.get('data')  # Get data from session
#         if data is None:
#             raise ValueError("Data not processed. Upload a file first.")
#         return data
#     except Exception as e:
#         logger.error(f"Error getting data: {e}")
#         raise


# @app.route('/regression', methods=['POST', 'OPTIONS'])
# def run_regression():
#     """Runs the selected regression model."""
#     logger.info(f"Received request to /regression")
#     logger.info(f"Request method: {request.method}")
#     logger.info(f"Request headers: {request.headers}")
#     logger.info(f"Request body: {request.get_data(as_text=True)}")  # Log the raw request body
#     try:
#         data = get_data()
#         model_name = request.json.get('model')
#         models_reg = {
#             "Linear Regression": LinearRegression(),
#             "Ridge Regression": Ridge(),
#             "Lasso Regression": Lasso(),
#             "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#             "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
#             "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
#             "Decision Tree": DecisionTreeRegressor(random_state=42),
#             "KNN": KNeighborsRegressor(n_neighbors=5),
#             "SVM": SVR(),
#             "Gaussian Process": GaussianProcessRegressor(kernel=RBF(), random_state=42)
#         }

#         if model_name not in models_reg:
#             logger.warning(f"Invalid regression model: {model_name}")
#             return jsonify({'error': 'Invalid regression model'}), 400

#         X_reg_train_scaled = data.get('X_reg_train_scaled')
#         X_reg_test_scaled = data.get('X_reg_test_scaled')
#         y_reg_train = data.get('y_reg_train')
#         y_reg_test = data.get('y_reg_test')

#         if X_reg_train_scaled is None or X_reg_test_scaled is None or y_reg_train is None or y_reg_test is None:
#             logger.error("Regression data not available. Ensure file is uploaded and processed correctly.")
#             return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#         model = models_reg[model_name]
#         model.fit(X_reg_train_scaled, y_reg_train)
#         y_pred = model.predict(X_reg_test_scaled)

#         results = {
#             "MAE": mean_absolute_error(y_reg_test, y_pred),
#             "MSE": mean_squared_error(y_reg_test, y_pred),
#             "R2 Score": r2_score(y_reg_test, y_pred)
#         }
#         logger.info(f"Regression model '{model_name}' ran successfully. Results: {results}")
#         response = jsonify(results)
#         response.headers['Content-Type'] = 'application/json'
#         return response, 200

#     except Exception as e:
#         logger.exception(f"Error running regression model: {e}")
#         return jsonify({'error': f"Error running regression: {str(e)}"}), 500


# @app.route('/classification', methods=['POST'])
# def run_classification():
#     """Runs the selected classification model."""
#     logger.info(f"Received request to /classification")
#     logger.info(f"Request method: {request.method}")
#     logger.info(f"Request headers: {request.headers}")
#     logger.info(f"Request body: {request.get_data(as_text=True)}")  # Log the raw request body

#     try:
#         data = get_data()
#         model_name = request.json.get('model')
#         models_cls = {
#             "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
#             "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#             "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
#             "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
#             "Decision Tree": DecisionTreeClassifier(random_state=42),
#             "KNN": KNeighborsClassifier(n_neighbors=5),
#             "SVM": SVC(probability=True),  # Enable probability for ROC AUC
#             "Gaussian Naive Bayes": GaussianNB(),
#             "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
#             "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
#             "Gaussian Process": GaussianProcessClassifier(kernel=RBF(), random_state=42),
#             "MLP Classifier": MLPClassifier(random_state=42)
#         }

#         if model_name not in models_cls:
#             logger.warning(f"Invalid classification model: {model_name}")
#             return jsonify({'error': 'Invalid classification model'}), 400

#         X_cls_train_scaled = data.get('X_cls_train_scaled')
#         X_cls_test_scaled = data.get('X_cls_test_scaled')
#         y_cls_train = data.get('y_cls_train')
#         y_cls_test = data.get('y_cls_test')
#         categorical_cols = data.get('categorical_cols')
#         label_encoders = data.get('label_encoders')

#         if X_cls_train_scaled is None or X_cls_test_scaled is None or y_cls_train is None or y_cls_test is None:
#             logger.error("Classification data not available. Ensure file is uploaded and processed correctly.")
#             return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#         model = models_cls[model_name]
#         model.fit(X_cls_train_scaled, y_cls_train)
#         y_pred = model.predict(X_cls_test_scaled)
#         y_pred_proba = model.predict_proba(X_cls_test_scaled)

#         metrics = {
#             "Accuracy": accuracy_score(y_cls_test, y_pred),
#             "Precision": precision_score(y_cls_test, y_pred, average='weighted', zero_division=0),
#             "Recall": recall_score(y_cls_test, y_pred, average='weighted', zero_division=0),
#             "F1 Score": f1_score(y_cls_test, y_pred, average='weighted', zero_division=0)
#         }

#         # ROC AUC (only if binary classification)
#         if len(np.unique(y_cls_train)) == 2:
#             try:
#                 metrics["ROC AUC"] = roc_auc_score(y_cls_test, y_pred_proba[:, 1])
#             except Exception as e:
#                 logger.warning(f"Error calculating ROC AUC: {e}")
#                 metrics["ROC AUC"] = "N/A"

#         # Generate and encode confusion matrix plot
#         cm = confusion_matrix(y_cls_test, y_pred)
#         plt.figure(figsize=(6, 4))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                     xticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols and label_encoders else None,
#                     yticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols and label_encoders else None)
#         plt.title(f'Confusion Matrix - {model_name}')
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')

#         # Save the plot to a buffer
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         image_base64 = base64.b64encode(buf.read()).decode('utf-8')
#         plt.close()

#         logger.info(f"Classification model '{model_name}' ran successfully. Metrics: {metrics}")
#         response = jsonify({'metrics': metrics, 'confusion_matrix': image_base64})
#         response.headers['Content-Type'] = 'application/json'
#         return response, 200

#     except Exception as e:
#         logger.exception(f"Error running classification model: {e}")
#         return jsonify({'error': f"Error running classification: {str(e)}"}), 500


# @app.route('/clustering', methods=['GET'])
# def run_clustering():
#     """Runs the clustering algorithm and returns analysis."""
#     logger.info(f"Received request to /clustering")
#     logger.info(f"Request method: {request.method}")
#     logger.info(f"Request headers: {request.headers}")
#     try:
#         data = get_data()
#         df = data.get('df')
#         numerical_cols = data.get('numerical_cols')

#         if df is None:
#             logger.error("Dataframe not available. Ensure file is uploaded and processed correctly.")
#             return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#         kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
#         df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

#         # Analyze cluster characteristics
#         cluster_analysis = df.groupby('Cluster')[numerical_cols].mean().to_dict('index')

#         # Generate cluster distribution plot
#         plt.figure(figsize=(8, 6))
#         sns.countplot(x='Cluster', data=df)
#         plt.title('Distribution of Clusters')
#         buf_cluster_dist = io.BytesIO()
#         plt.savefig(buf_cluster_dist, format='png')
#         buf_cluster_dist.seek(0)
#         cluster_dist_image_base64 = base64.b64encode(buf_cluster_dist.read()).decode('utf-8')
#         plt.close()

#         # Generate box plots for each numerical column
#         box_plot_images = {}
#         for col in numerical_cols:
#             plt.figure(figsize=(8, 6))
#             sns.boxplot(x='Cluster', y=col, data=df)
#             plt.title(f'{col} distribution across clusters')
#             buf_box = io.BytesIO()
#             plt.savefig(buf_box, format='png')
#             buf_box.seek(0)
#             box_plot_images[col] = base64.b64encode(buf_box.read()).decode('utf-8')
#             plt.close()

#         def get_most_distinctive_features(cluster_id, top_n=3):
#             cluster_data = df[df['Cluster'] == cluster_id][numerical_cols]
#             overall_mean = df[numerical_cols].mean()
#             cluster_mean = cluster_data.mean()
#             feature_importance = abs(cluster_mean - overall_mean)
#             most_important = feature_importance.nlargest(top_n)
#             return most_important.index.tolist()

#         distinctive_features = {}
#         for i in range(3):  # Assuming 3 clusters
#             distinctive_features[i] = get_most_distinctive_features(i)

#         result = {
#             'cluster_analysis': cluster_analysis,
#             'cluster_distribution_plot': cluster_dist_image_base64,
#             'box_plot_images': box_plot_images,
#             'distinctive_features': distinctive_features
#         }
#         logger.info(f"Clustering ran successfully.")
#         response = jsonify(result)
#         response.headers['Content-Type'] = 'application/json'
#         return response, 200

#     except Exception as e:
#         logger.exception(f"Error running clustering: {e}")
#         return jsonify({'error': str(e)}), 500


















# # app.py
# from flask import Flask, request, jsonify, session  # Import session
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.svm import SVR, SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
# from sklearn.cluster import KMeans
# import io
# import base64
# import matplotlib.pyplot as plt
# from flask_cors import CORS
# import seaborn as sns
# import os
# import logging
# import secrets  # For generating a secure secret key


# # Ensure matplotlib uses a non-GUI backend
# import matplotlib
# matplotlib.use('Agg')

# app = Flask(__name__)
# CORS(app)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Configure session
# secret_key = secrets.token_hex(16)  # Generate a random 32-character hex string
# app.secret_key = secret_key # Store in environment variable for production
# #app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem sessions (adjust for production) #You can also use redis or other database
# #Session(app)

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def process_data(file):
#     """Processes the uploaded data and prepares it for modeling."""
#     try:
#         df = pd.read_csv(file)
#         logger.info("Dataframe loaded successfully")

#         # Check for NaN values
#         if df.isnull().any().any():
#             nan_cols = df.columns[df.isnull().any()].tolist()
#             raise ValueError(f'NaN values found in columns: {", ".join(nan_cols)}')

#         # Drop columns with too many missing values
#         df = df.dropna(thresh=len(df) * 0.6, axis=1)

#         # Fill missing values
#         for col in df.select_dtypes(include=['number']).columns:
#             df[col].fillna(df[col].median(), inplace=True)
#         for col in df.select_dtypes(include=['object']).columns:
#             df[col].fillna(df[col].mode()[0], inplace=True)

#         # Identify numerical and categorical columns
#         numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
#         categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

#         # Encode categorical columns
#         label_encoders = {}
#         for col in categorical_cols:
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col].astype(str))
#             label_encoders[col] = le

#         # Select features and target dynamically
#         X_reg, y_reg = None, None
#         X_cls, y_cls = None, None
#         X_cluster = None

#         if len(numerical_cols) >= 2:
#             X_reg = df[numerical_cols[:-1]]  # All but last numeric column as features
#             y_reg = df[numerical_cols[-1]]   # Last numeric column as target

#         if len(numerical_cols) >= 3:
#             X_cls = df[numerical_cols[:-1]]  # All but last numeric column as features
#             y_cls = df[numerical_cols[-1]]   # Last numeric column as target

#         X_cluster = df[numerical_cols] if len(numerical_cols) >= 2 else None

#         X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test = None, None, None, None
#         X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test = None, None, None, None

#         # Train-test split
#         if X_reg is not None:
#             X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

#             # Standardization
#             scaler = StandardScaler()
#             X_reg_train_scaled = scaler.fit_transform(X_reg_train)
#             X_reg_test_scaled = scaler.transform(X_reg_test)

#         if X_cls is not None:
#             X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

#             # Standardization
#             scaler = StandardScaler()
#             X_cls_train_scaled = scaler.fit_transform(X_cls_train)
#             X_cls_test_scaled = scaler.transform(X_cls_test)
        
#         # Store processed data in the request context
#         data = {
#             'df': df,
#             'X_reg_train_scaled': X_reg_train_scaled,
#             'X_reg_test_scaled': X_reg_test_scaled,
#             'y_reg_train': y_reg_train,
#             'y_reg_test': y_reg_test,
#             'X_cls_train_scaled': X_cls_train_scaled,
#             'X_cls_test_scaled': X_cls_test_scaled,
#             'y_cls_train': y_cls_train,
#             'y_cls_test': y_cls_test,
#             'label_encoders': label_encoders,
#             'numerical_cols': numerical_cols,
#             'categorical_cols': categorical_cols
#         }
#         return data  # Return the processed data as a dictionary
    
#     except Exception as e:
#         logger.error(f"Error processing data: {e}")
#         raise


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     """Handles file upload and data processing."""
#     try:
#         if 'file' not in request.files:
#             logger.warning("No file part in the request")
#             return jsonify({'error': 'No file part'}), 400

#         file = request.files['file']
#         if file.filename == '':
#             logger.warning("No selected file")
#             return jsonify({'error': 'No selected file'}), 400

#         data = process_data(io.StringIO(file.stream.read().decode("UTF8", errors='ignore'), newline=None))  # Passes the file to process_data function
#         session['data'] = data # Storing processed data in session
#         return jsonify({'message': 'File uploaded and processed successfully'}), 200

#     except ValueError as e:
#         logger.error(f"Validation error: {e}")
#         return jsonify({'error': str(e)}), 400
#     except Exception as e:
#         logger.exception("An unexpected error occurred during file upload")
#         return jsonify({'error': str(e)}), 500


# def get_data():
#     """Helper function to retrieve processed data from the session."""
#     try:
#         data = session.get('data') # Get data from session
#         if data is None:
#             raise ValueError("Data not processed. Upload a file first.")
#         return data
#     except Exception as e:
#         logger.error(f"Error getting data: {e}")
#         raise


# @app.route('/regression', methods=['POST'])
# def run_regression():
#     """Runs the selected regression model."""
#     try:
#         data = get_data()
#         model_name = request.json.get('model')
#         models_reg = {
#             "Linear Regression": LinearRegression(),
#             "Ridge Regression": Ridge(),
#             "Lasso Regression": Lasso(),
#             "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#             "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
#             "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
#             "Decision Tree": DecisionTreeRegressor(random_state=42),
#             "KNN": KNeighborsRegressor(n_neighbors=5),
#             "SVM": SVR(),
#             "Gaussian Process": GaussianProcessRegressor(kernel=RBF(), random_state=42)
#         }

#         if model_name not in models_reg:
#             logger.warning(f"Invalid regression model: {model_name}")
#             return jsonify({'error': 'Invalid regression model'}), 400

#         X_reg_train_scaled = data.get('X_reg_train_scaled')
#         X_reg_test_scaled = data.get('X_reg_test_scaled')
#         y_reg_train = data.get('y_reg_train')
#         y_reg_test = data.get('y_reg_test')

#         if X_reg_train_scaled is None or X_reg_test_scaled is None or y_reg_train is None or y_reg_test is None:
#             logger.error("Regression data not available. Ensure file is uploaded and processed correctly.")
#             return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#         model = models_reg[model_name]
#         model.fit(X_reg_train_scaled, y_reg_train)
#         y_pred = model.predict(X_reg_test_scaled)

#         results = {
#             "MAE": mean_absolute_error(y_reg_test, y_pred),
#             "MSE": mean_squared_error(y_reg_test, y_pred),
#             "R2 Score": r2_score(y_reg_test, y_pred)
#         }
#         logger.info(f"Regression model '{model_name}' ran successfully. Results: {results}")
#         return jsonify(results), 200

#     except Exception as e:
#         logger.exception(f"Error running regression model: {e}")
#         return jsonify({'error': f"Error running regression: {str(e)}"}), 500


# @app.route('/classification', methods=['POST'])
# def run_classification():
#     """Runs the selected classification model."""
#     try:
#         data = get_data()
#         model_name = request.json.get('model')
#         models_cls = {
#             "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
#             "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#             "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
#             "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
#             "Decision Tree": DecisionTreeClassifier(random_state=42),
#             "KNN": KNeighborsClassifier(n_neighbors=5),
#             "SVM": SVC(probability=True),  # Enable probability for ROC AUC
#             "Gaussian Naive Bayes": GaussianNB(),
#             "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
#             "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
#             "Gaussian Process": GaussianProcessClassifier(kernel=RBF(), random_state=42),
#             "MLP Classifier": MLPClassifier(random_state=42)
#         }

#         if model_name not in models_cls:
#             logger.warning(f"Invalid classification model: {model_name}")
#             return jsonify({'error': 'Invalid classification model'}), 400

#         X_cls_train_scaled = data.get('X_cls_train_scaled')
#         X_cls_test_scaled = data.get('X_cls_test_scaled')
#         y_cls_train = data.get('y_cls_train')
#         y_cls_test = data.get('y_cls_test')
#         categorical_cols = data.get('categorical_cols')
#         label_encoders = data.get('label_encoders')

#         if X_cls_train_scaled is None or X_cls_test_scaled is None or y_cls_train is None or y_cls_test is None:
#             logger.error("Classification data not available. Ensure file is uploaded and processed correctly.")
#             return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#         model = models_cls[model_name]
#         model.fit(X_cls_train_scaled, y_cls_train)
#         y_pred = model.predict(X_cls_test_scaled)
#         y_pred_proba = model.predict_proba(X_cls_test_scaled)

#         metrics = {
#             "Accuracy": accuracy_score(y_cls_test, y_pred),
#             "Precision": precision_score(y_cls_test, y_pred, average='weighted', zero_division=0),
#             "Recall": recall_score(y_cls_test, y_pred, average='weighted', zero_division=0),
#             "F1 Score": f1_score(y_cls_test, y_pred, average='weighted', zero_division=0)
#         }

#         # ROC AUC (only if binary classification)
#         if len(np.unique(y_cls_train)) == 2:
#             try:
#                 metrics["ROC AUC"] = roc_auc_score(y_cls_test, y_pred_proba[:, 1])
#             except Exception as e:
#                 logger.warning(f"Error calculating ROC AUC: {e}")
#                 metrics["ROC AUC"] = "N/A"

#         # Generate and encode confusion matrix plot
#         cm = confusion_matrix(y_cls_test, y_pred)
#         plt.figure(figsize=(6, 4))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                     xticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols and label_encoders else None,
#                     yticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols and label_encoders else None)
#         plt.title(f'Confusion Matrix - {model_name}')
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')

#         # Save the plot to a buffer
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         image_base64 = base64.b64encode(buf.read()).decode('utf-8')
#         plt.close()

#         logger.info(f"Classification model '{model_name}' ran successfully. Metrics: {metrics}")
#         return jsonify({'metrics': metrics, 'confusion_matrix': image_base64}), 200

#     except Exception as e:
#         logger.exception(f"Error running classification model: {e}")
#         return jsonify({'error': f"Error running classification: {str(e)}"}), 500


# @app.route('/clustering', methods=['GET'])
# def run_clustering():
#     """Runs the clustering algorithm and returns analysis."""
#     try:
#         data = get_data()
#         df = data.get('df')
#         numerical_cols = data.get('numerical_cols')

#         if df is None:
#             logger.error("Dataframe not available. Ensure file is uploaded and processed correctly.")
#             return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#         kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
#         df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

#         # Analyze cluster characteristics
#         cluster_analysis = df.groupby('Cluster')[numerical_cols].mean().to_dict('index')

#         # Generate cluster distribution plot
#         plt.figure(figsize=(8, 6))
#         sns.countplot(x='Cluster', data=df)
#         plt.title('Distribution of Clusters')
#         buf_cluster_dist = io.BytesIO()
#         plt.savefig(buf_cluster_dist, format='png')
#         buf_cluster_dist.seek(0)
#         cluster_dist_image_base64 = base64.b64encode(buf_cluster_dist.read()).decode('utf-8')
#         plt.close()

#         # Generate box plots for each numerical column
#         box_plot_images = {}
#         for col in numerical_cols:
#             plt.figure(figsize=(8, 6))
#             sns.boxplot(x='Cluster', y=col, data=df)
#             plt.title(f'{col} distribution across clusters')
#             buf_box = io.BytesIO()
#             plt.savefig(buf_box, format='png')
#             buf_box.seek(0)
#             box_plot_images[col] = base64.b64encode(buf_box.read()).decode('utf-8')
#             plt.close()

#         def get_most_distinctive_features(cluster_id, top_n=3):
#             cluster_data = df[df['Cluster'] == cluster_id][numerical_cols]
#             overall_mean = df[numerical_cols].mean()
#             cluster_mean = cluster_data.mean()
#             feature_importance = abs(cluster_mean - overall_mean)
#             most_important = feature_importance.nlargest(top_n)
#             return most_important.index.tolist()

#         distinctive_features = {}
#         for i in range(3):  # Assuming 3 clusters
#             distinctive_features[i] = get_most_distinctive_features(i)

#         result = {
#             'cluster_analysis': cluster_analysis,
#             'cluster_distribution_plot': cluster_dist_image_base64,
#             'box_plot_images': box_plot_images,
#             'distinctive_features': distinctive_features
#         }
#         logger.info(f"Clustering ran successfully.")
#         return jsonify(result), 200

#     except Exception as e:
#         logger.exception(f"Error running clustering: {e}")
#         return jsonify({'error': str(e)}), 500





# app.py
# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.svm import SVR, SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
# from sklearn.cluster import KMeans
# import io
# import base64
# import matplotlib.pyplot as plt
# from flask_cors import CORS
# import seaborn as sns
# import os
# import logging  # Added for logging

# # Ensure matplotlib uses a non-GUI backend
# import matplotlib
# matplotlib.use('Agg')

# app = Flask(__name__)
# CORS(app)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Remove global variables and use request context for data
# # Store dataframe and processed data within the request context
# # Initialize request_context at the beginning of the request


# def process_data(file):
#     """Processes the uploaded data and prepares it for modeling."""
#     try:
#         df = pd.read_csv(file)
#         logger.info("Dataframe loaded successfully")

#         # Check for NaN values
#         if df.isnull().any().any():
#             nan_cols = df.columns[df.isnull().any()].tolist()
#             raise ValueError(f'NaN values found in columns: {", ".join(nan_cols)}')

#         # Drop columns with too many missing values
#         df = df.dropna(thresh=len(df) * 0.6, axis=1)

#         # Fill missing values
#         for col in df.select_dtypes(include=['number']).columns:
#             df[col].fillna(df[col].median(), inplace=True)
#         for col in df.select_dtypes(include=['object']).columns:
#             df[col].fillna(df[col].mode()[0], inplace=True)

#         # Identify numerical and categorical columns
#         numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
#         categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

#         # Encode categorical columns
#         label_encoders = {}
#         for col in categorical_cols:
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col].astype(str))
#             label_encoders[col] = le

#         # Select features and target dynamically
#         X_reg, y_reg = None, None
#         X_cls, y_cls = None, None
#         X_cluster = None

#         if len(numerical_cols) >= 2:
#             X_reg = df[numerical_cols[:-1]]  # All but last numeric column as features
#             y_reg = df[numerical_cols[-1]]   # Last numeric column as target

#         if len(numerical_cols) >= 3:
#             X_cls = df[numerical_cols[:-1]]  # All but last numeric column as features
#             y_cls = df[numerical_cols[-1]]   # Last numeric column as target

#         X_cluster = df[numerical_cols] if len(numerical_cols) >= 2 else None

#         X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test = None, None, None, None
#         X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test = None, None, None, None

#         # Train-test split
#         if X_reg is not None:
#             X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

#             # Standardization
#             scaler = StandardScaler()
#             X_reg_train_scaled = scaler.fit_transform(X_reg_train)
#             X_reg_test_scaled = scaler.transform(X_reg_test)

#         if X_cls is not None:
#             X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

#             # Standardization
#             scaler = StandardScaler()
#             X_cls_train_scaled = scaler.fit_transform(X_cls_train)
#             X_cls_test_scaled = scaler.transform(X_cls_test)
        
#         # Store processed data in the request context
#         data = {
#             'df': df,
#             'X_reg_train_scaled': X_reg_train_scaled,
#             'X_reg_test_scaled': X_reg_test_scaled,
#             'y_reg_train': y_reg_train,
#             'y_reg_test': y_reg_test,
#             'X_cls_train_scaled': X_cls_train_scaled,
#             'X_cls_test_scaled': X_cls_test_scaled,
#             'y_cls_train': y_cls_train,
#             'y_cls_test': y_cls_test,
#             'label_encoders': label_encoders,
#             'numerical_cols': numerical_cols,
#             'categorical_cols': categorical_cols
#         }
#         return data  # Return the processed data as a dictionary
    
#     except Exception as e:
#         logger.error(f"Error processing data: {e}")
#         raise


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     """Handles file upload and data processing."""
#     try:
#         if 'file' not in request.files:
#             logger.warning("No file part in the request")
#             return jsonify({'error': 'No file part'}), 400

#         file = request.files['file']
#         if file.filename == '':
#             logger.warning("No selected file")
#             return jsonify({'error': 'No selected file'}), 400

#         data = process_data(io.StringIO(file.stream.read().decode("UTF8", errors='ignore'), newline=None))  # Passes the file to process_data function
#         request.data_store = data
#         return jsonify({'message': 'File uploaded and processed successfully'}), 200

#     except ValueError as e:
#         logger.error(f"Validation error: {e}")
#         return jsonify({'error': str(e)}), 400
#     except Exception as e:
#         logger.exception("An unexpected error occurred during file upload")
#         return jsonify({'error': str(e)}), 500


# def get_data():
#     """Helper function to retrieve processed data from the request context."""
#     try:
#         data = getattr(request, 'data_store', None)
#         if data is None:
#             raise ValueError("Data not processed. Upload a file first.")
#         return data
#     except Exception as e:
#         logger.error(f"Error getting data: {e}")
#         raise


# @app.route('/regression', methods=['POST'])
# def run_regression():
#     """Runs the selected regression model."""
#     try:
#         data = get_data()
#         model_name = request.json.get('model')
#         models_reg = {
#             "Linear Regression": LinearRegression(),
#             "Ridge Regression": Ridge(),
#             "Lasso Regression": Lasso(),
#             "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#             "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
#             "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
#             "Decision Tree": DecisionTreeRegressor(random_state=42),
#             "KNN": KNeighborsRegressor(n_neighbors=5),
#             "SVM": SVR(),
#             "Gaussian Process": GaussianProcessRegressor(kernel=RBF(), random_state=42)
#         }

#         if model_name not in models_reg:
#             logger.warning(f"Invalid regression model: {model_name}")
#             return jsonify({'error': 'Invalid regression model'}), 400

#         X_reg_train_scaled = data.get('X_reg_train_scaled')
#         X_reg_test_scaled = data.get('X_reg_test_scaled')
#         y_reg_train = data.get('y_reg_train')
#         y_reg_test = data.get('y_reg_test')

#         if X_reg_train_scaled is None or X_reg_test_scaled is None or y_reg_train is None or y_reg_test is None:
#             logger.error("Regression data not available. Ensure file is uploaded and processed correctly.")
#             return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#         model = models_reg[model_name]
#         model.fit(X_reg_train_scaled, y_reg_train)
#         y_pred = model.predict(X_reg_test_scaled)

#         results = {
#             "MAE": mean_absolute_error(y_reg_test, y_pred),
#             "MSE": mean_squared_error(y_reg_test, y_pred),
#             "R2 Score": r2_score(y_reg_test, y_pred)
#         }
#         logger.info(f"Regression model '{model_name}' ran successfully. Results: {results}")
#         return jsonify(results), 200

#     except Exception as e:
#         logger.exception(f"Error running regression model: {e}")
#         return jsonify({'error': f"Error running regression: {str(e)}"}), 500


# @app.route('/classification', methods=['POST'])
# def run_classification():
#     """Runs the selected classification model."""
#     try:
#         data = get_data()
#         model_name = request.json.get('model')
#         models_cls = {
#             "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
#             "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#             "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
#             "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
#             "Decision Tree": DecisionTreeClassifier(random_state=42),
#             "KNN": KNeighborsClassifier(n_neighbors=5),
#             "SVM": SVC(probability=True),  # Enable probability for ROC AUC
#             "Gaussian Naive Bayes": GaussianNB(),
#             "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
#             "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
#             "Gaussian Process": GaussianProcessClassifier(kernel=RBF(), random_state=42),
#             "MLP Classifier": MLPClassifier(random_state=42)
#         }

#         if model_name not in models_cls:
#             logger.warning(f"Invalid classification model: {model_name}")
#             return jsonify({'error': 'Invalid classification model'}), 400

#         X_cls_train_scaled = data.get('X_cls_train_scaled')
#         X_cls_test_scaled = data.get('X_cls_test_scaled')
#         y_cls_train = data.get('y_cls_train')
#         y_cls_test = data.get('y_cls_test')
#         categorical_cols = data.get('categorical_cols')
#         label_encoders = data.get('label_encoders')

#         if X_cls_train_scaled is None or X_cls_test_scaled is None or y_cls_train is None or y_cls_test is None:
#             logger.error("Classification data not available. Ensure file is uploaded and processed correctly.")
#             return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#         model = models_cls[model_name]
#         model.fit(X_cls_train_scaled, y_cls_train)
#         y_pred = model.predict(X_cls_test_scaled)
#         y_pred_proba = model.predict_proba(X_cls_test_scaled)

#         metrics = {
#             "Accuracy": accuracy_score(y_cls_test, y_pred),
#             "Precision": precision_score(y_cls_test, y_pred, average='weighted', zero_division=0),
#             "Recall": recall_score(y_cls_test, y_pred, average='weighted', zero_division=0),
#             "F1 Score": f1_score(y_cls_test, y_pred, average='weighted', zero_division=0)
#         }

#         # ROC AUC (only if binary classification)
#         if len(np.unique(y_cls_train)) == 2:
#             try:
#                 metrics["ROC AUC"] = roc_auc_score(y_cls_test, y_pred_proba[:, 1])
#             except Exception as e:
#                 logger.warning(f"Error calculating ROC AUC: {e}")
#                 metrics["ROC AUC"] = "N/A"

#         # Generate and encode confusion matrix plot
#         cm = confusion_matrix(y_cls_test, y_pred)
#         plt.figure(figsize=(6, 4))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                     xticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols and label_encoders else None,
#                     yticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols and label_encoders else None)
#         plt.title(f'Confusion Matrix - {model_name}')
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')

#         # Save the plot to a buffer
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         image_base64 = base64.b64encode(buf.read()).decode('utf-8')
#         plt.close()

#         logger.info(f"Classification model '{model_name}' ran successfully. Metrics: {metrics}")
#         return jsonify({'metrics': metrics, 'confusion_matrix': image_base64}), 200

#     except Exception as e:
#         logger.exception(f"Error running classification model: {e}")
#         return jsonify({'error': f"Error running classification: {str(e)}"}), 500


# @app.route('/clustering', methods=['GET'])
# def run_clustering():
#     """Runs the clustering algorithm and returns analysis."""
#     try:
#         data = get_data()
#         df = data.get('df')
#         numerical_cols = data.get('numerical_cols')

#         if df is None:
#             logger.error("Dataframe not available. Ensure file is uploaded and processed correctly.")
#             return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#         kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
#         df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

#         # Analyze cluster characteristics
#         cluster_analysis = df.groupby('Cluster')[numerical_cols].mean().to_dict('index')

#         # Generate cluster distribution plot
#         plt.figure(figsize=(8, 6))
#         sns.countplot(x='Cluster', data=df)
#         plt.title('Distribution of Clusters')
#         buf_cluster_dist = io.BytesIO()
#         plt.savefig(buf_cluster_dist, format='png')
#         buf_cluster_dist.seek(0)
#         cluster_dist_image_base64 = base64.b64encode(buf_cluster_dist.read()).decode('utf-8')
#         plt.close()

#         # Generate box plots for each numerical column
#         box_plot_images = {}
#         for col in numerical_cols:
#             plt.figure(figsize=(8, 6))
#             sns.boxplot(x='Cluster', y=col, data=df)
#             plt.title(f'{col} distribution across clusters')
#             buf_box = io.BytesIO()
#             plt.savefig(buf_box, format='png')
#             buf_box.seek(0)
#             box_plot_images[col] = base64.b64encode(buf_box.read()).decode('utf-8')
#             plt.close()

#         def get_most_distinctive_features(cluster_id, top_n=3):
#             cluster_data = df[df['Cluster'] == cluster_id][numerical_cols]
#             overall_mean = df[numerical_cols].mean()
#             cluster_mean = cluster_data.mean()
#             feature_importance = abs(cluster_mean - overall_mean)
#             most_important = feature_importance.nlargest(top_n)
#             return most_important.index.tolist()

#         distinctive_features = {}
#         for i in range(3):  # Assuming 3 clusters
#             distinctive_features[i] = get_most_distinctive_features(i)

#         result = {
#             'cluster_analysis': cluster_analysis,
#             'cluster_distribution_plot': cluster_dist_image_base64,
#             'box_plot_images': box_plot_images,
#             'distinctive_features': distinctive_features
#         }
#         logger.info(f"Clustering ran successfully.")
#         return jsonify(result), 200

#     except Exception as e:
#         logger.exception(f"Error running clustering: {e}")
#         return jsonify({'error': str(e)}), 500






#working perfectly
# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np  # Important for handling potential NaNs
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, AdaBoostRegressor, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.svm import SVR, SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
# from sklearn.cluster import KMeans
# import io  # For handling file uploads from frontend
# import base64

# import matplotlib.pyplot as plt
# from flask_cors import CORS
# import seaborn as sns
# import os

# # Ensure matplotlib uses a non-GUI backend for server environments
# import matplotlib
# matplotlib.use('Agg')


# app = Flask(__name__)
# CORS(app)
# app.config['UPLOAD_FOLDER'] = 'uploads'  # Create an uploads folder
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Global variables to store the data and models (for simplicity)
# df = None
# X_reg_train_scaled = None
# X_reg_test_scaled = None
# y_reg_train = None
# y_reg_test = None
# X_cls_train_scaled = None
# X_cls_test_scaled = None
# y_cls_train = None
# y_cls_test = None
# label_encoders = {}
# numerical_cols = []
# categorical_cols = []


# def process_data(file):
#     """Processes the uploaded data and prepares it for modeling.  This is MOST of your original notebook code."""
#     global df, X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test, X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test, label_encoders, numerical_cols, categorical_cols

#     df = pd.read_csv(file)

#      # Check for NaN values *before* any processing
#     if df.isnull().any().any():
#         nan_cols = df.columns[df.isnull().any()].tolist()
#         raise ValueError(f'NaN values found in columns: {", ".join(nan_cols)}')


#     # Drop columns with too many missing values
#     df = df.dropna(thresh=len(df) * 0.6, axis=1)

#     # Fill missing values
#     for col in df.select_dtypes(include=['number']).columns:
#         df[col].fillna(df[col].median(), inplace=True)
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col].fillna(df[col].mode()[0], inplace=True)

#     # Identify numerical and categorical columns
#     numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
#     categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

#     # Encode categorical columns
#     label_encoders = {}
#     for col in categorical_cols:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col].astype(str))
#         label_encoders[col] = le

#     # Select features and target dynamically
#     if len(numerical_cols) >= 2:
#         X_reg = df[numerical_cols[:-1]]  # All but last numeric column as features
#         y_reg = df[numerical_cols[-1]]   # Last numeric column as target
#     else:
#         X_reg, y_reg = None, None

#     if len(numerical_cols) >= 3:
#         X_cls = df[numerical_cols[:-1]]  # All but last numeric column as features
#         y_cls = df[numerical_cols[-1]]   # Last numeric column as target
#     else:
#         X_cls, y_cls = None, None

#     X_cluster = df[numerical_cols] if len(numerical_cols) >= 2 else None

#     # Train-test split
#     if X_reg is not None:
#         X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
#     if X_cls is not None:
#         X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

#     # Standardization
#     scaler = StandardScaler()
#     if X_reg is not None:
#         X_reg_train_scaled = scaler.fit_transform(X_reg_train)
#         X_reg_test_scaled = scaler.transform(X_reg_test)
#     if X_cls is not None:
#         X_cls_train_scaled = scaler.fit_transform(X_cls_train)
#         X_cls_test_scaled = scaler.transform(X_cls_test)


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     """Handles file upload and data processing."""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         process_data(io.StringIO(file.stream.read().decode("UTF8"), newline=None))  # Passes the file to process_data function
#         return jsonify({'message': 'File uploaded and processed successfully'}), 200
#     except ValueError as e:
#          return jsonify({'error': str(e)}), 400
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# @app.route('/regression', methods=['POST'])
# def run_regression():
#     """Runs the selected regression model."""
#     model_name = request.json.get('model')
#     models_reg = {
#         "Linear Regression": LinearRegression(),
#         "Ridge Regression": Ridge(),
#         "Lasso Regression": Lasso(),
#         "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
#         "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
#         "Decision Tree": DecisionTreeRegressor(random_state=42),
#         "KNN": KNeighborsRegressor(n_neighbors=5),
#         "SVM": SVR(),
#         "Gaussian Process": GaussianProcessRegressor(kernel=RBF(), random_state=42)
#     }

#     if model_name not in models_reg:
#         return jsonify({'error': 'Invalid regression model'}), 400

#     global X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test

#     if X_reg_train_scaled is None or X_reg_test_scaled is None or y_reg_train is None or y_reg_test is None:
#         return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#     model = models_reg[model_name]
#     try:
#         model.fit(X_reg_train_scaled, y_reg_train)
#         y_pred = model.predict(X_reg_test_scaled)

#         results = {
#             "MAE": mean_absolute_error(y_reg_test, y_pred),
#             "MSE": mean_squared_error(y_reg_test, y_pred),
#             "R2 Score": r2_score(y_reg_test, y_pred)
#         }
#         return jsonify(results), 200

#     except Exception as e:
#         return jsonify({'error': f"Error running {model_name}: {str(e)}"}), 500


# @app.route('/classification', methods=['POST'])
# def run_classification():
#     """Runs the selected classification model."""
#     model_name = request.json.get('model')
#     models_cls = {
#         "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
#         "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#         "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
#         "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
#         "Decision Tree": DecisionTreeClassifier(random_state=42),
#         "KNN": KNeighborsClassifier(n_neighbors=5),
#         "SVM": SVC(probability=True),  # Enable probability for ROC AUC
#         "Gaussian Naive Bayes": GaussianNB(),
#         "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
#         "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
#         "Gaussian Process": GaussianProcessClassifier(kernel=RBF(), random_state=42),
#         "MLP Classifier": MLPClassifier(random_state=42) # Neural Network
#     }

#     if model_name not in models_cls:
#         return jsonify({'error': 'Invalid classification model'}), 400

#     global X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test, categorical_cols, label_encoders

#     if X_cls_train_scaled is None or X_cls_test_scaled is None or y_cls_train is None or y_cls_test is None:
#         return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#     model = models_cls[model_name]
#     try:
#         model.fit(X_cls_train_scaled, y_cls_train)
#         y_pred = model.predict(X_cls_test_scaled)
#         y_pred_proba = model.predict_proba(X_cls_test_scaled)  # Get probabilities for ROC AUC

#         metrics = {
#             "Accuracy": accuracy_score(y_cls_test, y_pred),
#             "Precision": precision_score(y_cls_test, y_pred, average='weighted', zero_division=0),
#             "Recall": recall_score(y_cls_test, y_pred, average='weighted', zero_division=0),
#             "F1 Score": f1_score(y_cls_test, y_pred, average='weighted', zero_division=0)
#         }

#         # ROC AUC (only if binary classification)
#         if len(np.unique(y_cls_train)) == 2:
#             try:
#                 metrics["ROC AUC"] = roc_auc_score(y_cls_test, y_pred_proba[:, 1])
#             except Exception as e:
#                 print(f"Error calculating ROC AUC: {e}")
#                 metrics["ROC AUC"] = "N/A"  # Or some other placeholder

#         # Generate and encode confusion matrix plot
#         cm = confusion_matrix(y_cls_test, y_pred)
#         plt.figure(figsize=(6, 4))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                     xticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols else None,
#                     yticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols else None)
#         plt.title(f'Confusion Matrix - {model_name}')
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')

#         # Save the plot to a buffer
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         image_base64 = base64.b64encode(buf.read()).decode('utf-8')
#         plt.close() # Close the plot to free memory

#         return jsonify({'metrics': metrics, 'confusion_matrix': image_base64}), 200

#     except Exception as e:
#         return jsonify({'error': f"Error running {model_name}: {str(e)}"}), 500


# @app.route('/clustering', methods=['GET'])
# def run_clustering():
#     """Runs the clustering algorithm and returns analysis."""
#     global df, numerical_cols

#     if df is None:
#         return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#     try:
#         kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Setting n_init explicitly avoids a warning
#         df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

#         # Analyze cluster characteristics
#         cluster_analysis = df.groupby('Cluster')[numerical_cols].mean().to_dict('index')

#         # Generate cluster distribution plot
#         plt.figure(figsize=(8, 6))
#         sns.countplot(x='Cluster', data=df)
#         plt.title('Distribution of Clusters')
#         buf_cluster_dist = io.BytesIO()
#         plt.savefig(buf_cluster_dist, format='png')
#         buf_cluster_dist.seek(0)
#         cluster_dist_image_base64 = base64.b64encode(buf_cluster_dist.read()).decode('utf-8')
#         plt.close()

#         # Generate box plots for each numerical column
#         box_plot_images = {}
#         for col in numerical_cols:
#             plt.figure(figsize=(8, 6))
#             sns.boxplot(x='Cluster', y=col, data=df)
#             plt.title(f'{col} distribution across clusters')
#             buf_box = io.BytesIO()
#             plt.savefig(buf_box, format='png')
#             buf_box.seek(0)
#             box_plot_images[col] = base64.b64encode(buf_box.read()).decode('utf-8')
#             plt.close()

#         # Most distinctive features (as before)
#         def get_most_distinctive_features(cluster_id, top_n=3):
#             cluster_data = df[df['Cluster'] == cluster_id][numerical_cols]
#             overall_mean = df[numerical_cols].mean()
#             cluster_mean = cluster_data.mean()
#             feature_importance = abs(cluster_mean - overall_mean)
#             most_important = feature_importance.nlargest(top_n)
#             return most_important.index.tolist()

#         distinctive_features = {}
#         for i in range(3):  # Assuming 3 clusters
#             distinctive_features[i] = get_most_distinctive_features(i)


#         return jsonify({
#             'cluster_analysis': cluster_analysis,
#             'cluster_distribution_plot': cluster_dist_image_base64,
#             'box_plot_images': box_plot_images,
#             'distinctive_features': distinctive_features
#         }), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True)  # Remove debug=True for production


# working
# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np  # Important for handling potential NaNs
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.svm import SVR, SVC
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.cluster import KMeans
# import io  # For handling file uploads from frontend
# import base64
# import matplotlib.pyplot as plt
# from flask_cors import CORS
# import seaborn as sns
# import os

# # Ensure matplotlib uses a non-GUI backend for server environments
# import matplotlib
# matplotlib.use('Agg')


# app = Flask(__name__)
# CORS(app)
# app.config['UPLOAD_FOLDER'] = 'uploads'  # Create an uploads folder
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Global variables to store the data and models (for simplicity)
# df = None
# X_reg_train_scaled = None
# X_reg_test_scaled = None
# y_reg_train = None
# y_reg_test = None
# X_cls_train_scaled = None
# X_cls_test_scaled = None
# y_cls_train = None
# y_cls_test = None
# label_encoders = {}
# numerical_cols = []
# categorical_cols = []


# def process_data(file):
#     """Processes the uploaded data and prepares it for modeling.  This is MOST of your original notebook code."""
#     global df, X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test, X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test, label_encoders, numerical_cols, categorical_cols

#     df = pd.read_csv(file)

#      # Check for NaN values *before* any processing
#     if df.isnull().any().any():
#         nan_cols = df.columns[df.isnull().any()].tolist()
#         raise ValueError(f'NaN values found in columns: {", ".join(nan_cols)}')


#     # Drop columns with too many missing values
#     df = df.dropna(thresh=len(df) * 0.6, axis=1)

#     # Fill missing values
#     for col in df.select_dtypes(include=['number']).columns:
#         df[col].fillna(df[col].median(), inplace=True)
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col].fillna(df[col].mode()[0], inplace=True)

#     # Identify numerical and categorical columns
#     numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
#     categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

#     # Encode categorical columns
#     label_encoders = {}
#     for col in categorical_cols:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col].astype(str))
#         label_encoders[col] = le

#     # Select features and target dynamically
#     if len(numerical_cols) >= 2:
#         X_reg = df[numerical_cols[:-1]]  # All but last numeric column as features
#         y_reg = df[numerical_cols[-1]]   # Last numeric column as target
#     else:
#         X_reg, y_reg = None, None

#     if len(numerical_cols) >= 3:
#         X_cls = df[numerical_cols[:-1]]  # All but last numeric column as features
#         y_cls = df[numerical_cols[-1]]   # Last numeric column as target
#     else:
#         X_cls, y_cls = None, None

#     X_cluster = df[numerical_cols] if len(numerical_cols) >= 2 else None

#     # Train-test split
#     if X_reg is not None:
#         X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
#     if X_cls is not None:
#         X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

#     # Standardization
#     scaler = StandardScaler()
#     if X_reg is not None:
#         X_reg_train_scaled = scaler.fit_transform(X_reg_train)
#         X_reg_test_scaled = scaler.transform(X_reg_test)
#     if X_cls is not None:
#         X_cls_train_scaled = scaler.fit_transform(X_cls_train)
#         X_cls_test_scaled = scaler.transform(X_cls_test)


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     """Handles file upload and data processing."""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         process_data(io.StringIO(file.stream.read().decode("UTF8"), newline=None))  # Passes the file to process_data function
#         return jsonify({'message': 'File uploaded and processed successfully'}), 200
#     except ValueError as e:
#          return jsonify({'error': str(e)}), 400
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# @app.route('/regression', methods=['POST'])
# def run_regression():
#     """Runs the selected regression model."""
#     model_name = request.json.get('model')
#     models_reg = {
#         "Linear Regression": LinearRegression(),
#         "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
#         "KNN": KNeighborsRegressor(n_neighbors=10),
#         "SVM": SVR(kernel='rbf', C=1.0, gamma='scale')
#     }

#     if model_name not in models_reg:
#         return jsonify({'error': 'Invalid regression model'}), 400

#     global X_reg_train_scaled, X_reg_test_scaled, y_reg_train, y_reg_test

#     if X_reg_train_scaled is None or X_reg_test_scaled is None or y_reg_train is None or y_reg_test is None:
#         return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#     model = models_reg[model_name]
#     model.fit(X_reg_train_scaled, y_reg_train)
#     y_pred = model.predict(X_reg_test_scaled)

#     results = {
#         "MAE": mean_absolute_error(y_reg_test, y_pred),
#         "MSE": mean_squared_error(y_reg_test, y_pred),
#         "R2 Score": r2_score(y_reg_test, y_pred)
#     }
#     return jsonify(results), 200


# @app.route('/classification', methods=['POST'])
# def run_classification():
#     """Runs the selected classification model."""
#     model_name = request.json.get('model')
#     models_cls = {
#         "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
#         "KNN": KNeighborsClassifier(n_neighbors=10),
#         "SVM": SVC(kernel='rbf', C=1.0, gamma='scale')
#     }

#     if model_name not in models_cls:
#         return jsonify({'error': 'Invalid classification model'}), 400

#     global X_cls_train_scaled, X_cls_test_scaled, y_cls_train, y_cls_test, categorical_cols, label_encoders

#     if X_cls_train_scaled is None or X_cls_test_scaled is None or y_cls_train is None or y_cls_test is None:
#         return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#     model = models_cls[model_name]
#     model.fit(X_cls_train_scaled, y_cls_train)
#     y_pred = model.predict(X_cls_test_scaled)

#     metrics = {
#         "Accuracy": accuracy_score(y_cls_test, y_pred),
#         "Precision": precision_score(y_cls_test, y_pred, average='weighted', zero_division=0),
#         "Recall": recall_score(y_cls_test, y_pred, average='weighted', zero_division=0),
#         "F1 Score": f1_score(y_cls_test, y_pred, average='weighted', zero_division=0)
#     }

#     # Generate and encode confusion matrix plot
#     cm = confusion_matrix(y_cls_test, y_pred)
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols else None,
#                 yticklabels=label_encoders[categorical_cols[-1]].classes_ if categorical_cols else None)
#     plt.title(f'Confusion Matrix - {model_name}')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')

#     # Save the plot to a buffer
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     image_base64 = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close() # Close the plot to free memory

#     return jsonify({'metrics': metrics, 'confusion_matrix': image_base64}), 200


# @app.route('/clustering', methods=['GET'])
# def run_clustering():
#     """Runs the clustering algorithm and returns analysis."""
#     global df, numerical_cols

#     if df is None:
#         return jsonify({'error': 'Data not processed. Upload a file first.'}), 400

#     try:
#         kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Setting n_init explicitly avoids a warning
#         df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

#         # Analyze cluster characteristics
#         cluster_analysis = df.groupby('Cluster')[numerical_cols].mean().to_dict('index')

#         # Generate cluster distribution plot
#         plt.figure(figsize=(8, 6))
#         sns.countplot(x='Cluster', data=df)
#         plt.title('Distribution of Clusters')
#         buf_cluster_dist = io.BytesIO()
#         plt.savefig(buf_cluster_dist, format='png')
#         buf_cluster_dist.seek(0)
#         cluster_dist_image_base64 = base64.b64encode(buf_cluster_dist.read()).decode('utf-8')
#         plt.close()

#         # Generate box plots for each numerical column
#         box_plot_images = {}
#         for col in numerical_cols:
#             plt.figure(figsize=(8, 6))
#             sns.boxplot(x='Cluster', y=col, data=df)
#             plt.title(f'{col} distribution across clusters')
#             buf_box = io.BytesIO()
#             plt.savefig(buf_box, format='png')
#             buf_box.seek(0)
#             box_plot_images[col] = base64.b64encode(buf_box.read()).decode('utf-8')
#             plt.close()

#         # Most distinctive features (as before)
#         def get_most_distinctive_features(cluster_id, top_n=3):
#             cluster_data = df[df['Cluster'] == cluster_id][numerical_cols]
#             overall_mean = df[numerical_cols].mean()
#             cluster_mean = cluster_data.mean()
#             feature_importance = abs(cluster_mean - overall_mean)
#             most_important = feature_importance.nlargest(top_n)
#             return most_important.index.tolist()

#         distinctive_features = {}
#         for i in range(3):  # Assuming 3 clusters
#             distinctive_features[i] = get_most_distinctive_features(i)


#         return jsonify({
#             'cluster_analysis': cluster_analysis,
#             'cluster_distribution_plot': cluster_dist_image_base64,
#             'box_plot_images': box_plot_images,
#             'distinctive_features': distinctive_features
#         }), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True)  # Remove debug=True for production
