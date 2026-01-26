# Machine Learning Project

A comprehensive machine learning project implementing **K-Nearest Neighbors (KNN)**, **Decision Tree**, and **K-Means Clustering** algorithms with complete data preprocessing, evaluation, and visualization pipelines.

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Notebooks](#notebooks)
- [Testing](#testing)
- [Results](#results)
- [Documentation](#documentation)

## 📁 Project Structure

```
ml-project/
├── data/
│   ├── raw/              # Raw data files
│   ├── processed/        # Processed data files
│   └── results/          # Model results and visualizations
├── notebooks/
│   ├── 01_exploration.ipynb      # Data exploration
│   ├── 02_preprocessing.ipynb    # Data preprocessing
│   └── 03_modeling.ipynb         # Model training and evaluation
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py      # Data loading, cleaning, scaling, splitting
│   ├── models/
│   │   ├── __init__.py
│   │   ├── knn.py               # KNN classifier
│   │   ├── decision_tree.py     # Decision Tree classifier
│   │   └── kmeans.py            # K-Means clustering
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # Evaluation metrics
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py             # Visualization functions
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_models.py           # Unit tests
├── main.py                      # Main pipeline
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## ✨ Features

### Models
- **K-Nearest Neighbors (KNN)**
  - Multiple distance metrics (Euclidean, Manhattan, Minkowski)
  - Configurable number of neighbors
  - Weighted and uniform voting

- **Decision Tree Classifier**
  - Gini and Entropy criteria
  - Configurable tree depth and splitting parameters
  - Feature importance extraction

- **K-Means Clustering**
  - Elbow method for optimal cluster selection
  - Configurable number of clusters
  - Cluster center visualization

### Data Processing
- Data loading from multiple formats (CSV, Excel, JSON)
- Missing value handling (drop, fill, mean, median)
- Duplicate removal
- Multiple scaling methods (Standard, MinMax, Robust)
- Stratified train-test splitting

### Evaluation
- Classification metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Clustering metrics: Silhouette score
- Comprehensive classification reports

### Visualization
- Decision boundary plots
- Confusion matrix heatmaps
- Elbow curve plots
- Cluster visualizations
- Feature importance plots
- Learning curves

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd data_ml_project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Main Pipeline

The main pipeline executes all models and generates results:

```bash
python main.py
```

This will:
1. Load and preprocess the sample Iris dataset
2. Train KNN and Decision Tree classifiers
3. Perform K-Means clustering with elbow method
4. Evaluate all models
5. Generate visualizations
6. Save results to `data/results/`

### Using Individual Models

#### KNN Classifier
```python
from src.models.knn import KNNClassifier

knn = KNNClassifier(n_neighbors=5, distance_metric='euclidean')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
```

#### Decision Tree Classifier
```python
from src.models.decision_tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini', max_depth=5)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
importances = dt.get_feature_importance()
```

#### K-Means Clustering
```python
from src.models.kmeans import KMeansClustering

kmeans = KMeansClustering(n_clusters=3)
labels = kmeans.fit_predict(X)
centers = kmeans.get_cluster_centers()

# Elbow method
k_values, inertias = kmeans.elbow_method(X, k_range=[2, 3, 4, 5])
```

### Data Preprocessing
```python
from src.data.preprocessing import preprocess_pipeline

processed_data = preprocess_pipeline(
    filepath='data/raw/your_data.csv',
    target_column='target',
    test_size=0.2,
    scaling_method='standard',
    random_state=42
)

X_train = processed_data['X_train']
X_test = processed_data['X_test']
y_train = processed_data['y_train']
y_test = processed_data['y_test']
```

## ⚙️ Configuration

Edit `config.yaml` to customize model hyperparameters:

```yaml
models:
  knn:
    n_neighbors: 5
    distance_metric: "euclidean"
    weights: "uniform"
    
  decision_tree:
    criterion: "gini"
    max_depth: 10
    min_samples_split: 2
    
  kmeans:
    n_clusters: 3
    max_iter: 300
    elbow_range: [2, 11]

preprocessing:
  test_size: 0.2
  scaling_method: "standard"
  random_state: 42
```

## 📓 Notebooks

The project includes three Jupyter notebooks:

1. **01_exploration.ipynb**: Data exploration and visualization
   - Dataset overview
   - Statistical analysis
   - Distribution plots
   - Correlation analysis

2. **02_preprocessing.ipynb**: Data preprocessing pipeline
   - Data cleaning
   - Feature scaling comparison
   - Train-test splitting
   - Data saving

3. **03_modeling.ipynb**: Model training and evaluation
   - KNN with different k values
   - Decision Tree with different criteria
   - K-Means with elbow method
   - Model comparison

To run the notebooks:
```bash
jupyter notebook
```

## 🧪 Testing

Run unit tests with pytest:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v
```

Test coverage includes:
- Model initialization and validation
- Fit and predict functionality
- Error handling
- Evaluation metrics
- Edge cases

## 📊 Results

After running the main pipeline, results are saved to `data/results/`:

- `knn_confusion_matrix.png` - KNN confusion matrix
- `dt_confusion_matrix.png` - Decision Tree confusion matrix
- `dt_feature_importance.png` - Feature importance plot
- `kmeans_elbow_curve.png` - Elbow method plot
- `kmeans_clusters.png` - Cluster visualization
- `model_comparison.csv` - Model performance comparison

## 📚 Documentation

### Module Documentation

All modules include comprehensive docstrings with:
- Function/class descriptions
- Parameter specifications
- Return value descriptions
- Error handling information
- Usage examples

### Error Handling

The project implements robust error handling:
- Input validation
- Type checking
- Informative error messages
- Logging throughout the pipeline

### Logging

Logging is configured throughout the project:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

Logs include:
- Data loading and preprocessing steps
- Model training progress
- Evaluation results
- Error messages and warnings

## 🔧 Extending the Project

### Adding New Models

1. Create a new file in `src/models/`
2. Implement the model class with `fit()`, `predict()`, and `score()` methods
3. Add the model to `src/models/__init__.py`
4. Update `main.py` to include the new model

### Adding New Datasets

1. Place data in `data/raw/`
2. Update `config.yaml` with data path
3. Modify `main.py` to load your dataset
4. Adjust preprocessing parameters as needed

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Happy Machine Learning! 🚀**
