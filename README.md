# 🧠 **Epilepsy MLOps Project**

Welcome to the **Epilepsy MLOps Project**, a robust and scalable pipeline designed for end-to-end machine learning operations (MLOps). This repository is structured to handle everything from data preprocessing to model deployment, ensuring reproducibility, scalability, and security. Below is an overview of the project's architecture, key features, and recommendations for improvement.

---

## 📂 **Repository Structure**

```
├── .dvc                 # DVC tracking files for versioning datasets and models
├── .dvcignore           # Files ignored by DVC
├── README.md            # Project documentation
├── docker-compose.yml   # Docker Compose configuration for container orchestration
├── metrics.dvc          # Versioned metrics for model evaluation
├── models.dvc           # Tracked versions of trained models
├── pipeline.py          # Prefect-based pipeline for automating workflows
├── production.dvc       # Tracks the current production-ready model
├── requirements.txt     # Global dependencies for the project
├── data/                # Data directory with raw, processed, and inference-specific datasets
│   ├── patients.dvc
│   ├── patients_inference.dvc
│   ├── processed.dvc
│   └── raw.dvc
├── mlflow_data/         # MLflow database and artifacts for experiment tracking
│   └── mlflow.db
├── services/            # Microservices for different components of the pipeline
│   ├── authentication/  # JWT-based authentication API
│   ├── evaluate/        # Model evaluation and promotion scripts
│   ├── inference/       # FastAPI-based inference API
│   ├── model_training/  # Model training service
│   ├── patient_data_pull/ # Extracts patient-specific data for inference
│   └── preprocessing/   # Preprocessing service for data cleaning and preparation
└── .gitignore           # Files ignored by Git
```

---

## 🔧 **Key Features**

### 1. **Infrastructure MLOps**

- **Docker Compose**:
  - Manages multiple services in isolated containers, ensuring dependency isolation and ease of deployment.
  - Services include MLflow for experiment tracking, APIs for authentication and inference, and microservices for preprocessing, training, and evaluation.
- **Microservice Architecture**:
  - Each service (e.g., preprocessing, training, inference) is containerized, making it modular, testable, and deployable independently.

### 2. **Data Management with DVC**

- **Version Control for Data**:
  - Uses [DVC](https://dvc.org/) to track changes in datasets, ensuring reproducibility and consistency across experiments.
  - `.dvc` files manage large datasets efficiently, integrating seamlessly with Git workflows.
- **Remote Storage**:
  - Configured with Dagshub S3 for remote storage, enabling collaboration and backup of datasets and models.

### 3. **Preprocessing**

- **Balanced Sampling**:
  - Ensures equal representation of classes in the training dataset, addressing class imbalance issues.
- **LSTM-Compatible Format**:
  - Processes raw data into `.npy` files optimized for LSTM input, improving efficiency during model training.

### 4. **Model Training**

- **LSTM Architecture**:
  - A two-layer LSTM followed by dense layers is used for temporal sequence modeling, ideal for epilepsy prediction tasks.
- **MLflow Integration**:
  - Tracks experiments, hyperparameters, and metrics for each training run, ensuring transparency and reproducibility.

### 5. **Evaluation and Promotion**

- **Automatic Model Evaluation**:
  - Compares new models against existing ones based on `val_accuracy`, promoting only the best-performing model to production.
- **Versioned Production Models**:
  - Tracks the production-ready model using `production.dvc`, ensuring traceability and accountability.

### 6. **Inference API**

- **FastAPI-Based Endpoint**:
  - Provides real-time predictions via `/predict/{patient_id}`.
  - Secured with JWT-based authentication to restrict unauthorized access.
- **Dynamic Model Reloading**:
  - Includes a `/reload-model` endpoint to update the deployed model without downtime.

### 7. **Security with JWT Authentication**

- **Token-Based Access Control**:
  - Protects sensitive endpoints with JWT tokens, ensuring secure access to APIs.
- **Future Improvements**:
  - Add token expiration management and automatic renewal mechanisms for enhanced security.

---

## 🚀 **Getting Started**

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- DVC installed (`pip install dvc`)
- MLflow server running (`docker-compose up -d mlflow`)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/epilepsy_mlops.git
   cd epilepsy_mlops
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Pull Raw Data**:

   ```bash
   dvc pull data/raw.dvc -r origin
   ```

4. **Build Docker Images**:
   ```bash
   docker-compose build
   ```

### Running the Pipeline

1. **Start MLflow Server**:

   ```bash
   docker-compose up -d mlflow
   ```

2. **Run Preprocessing**:

   ```bash
   docker-compose run --rm preprocessing
   ```

3. **Train the Model**:

   ```bash
   docker-compose run --rm model_training
   ```

4. **Evaluate and Promote the Model**:

   ```bash
   docker-compose run --rm evaluate
   ```

5. **Launch APIs**:

   ```bash
   docker-compose up -d auth_api inference_api
   ```

6. **Test Inference**:
   - Obtain a JWT token:
     ```bash
     curl -X POST "http://localhost:8000/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "alice", "password": "secret"}'
     ```
   - Use the token to make predictions:
     ```bash
     curl -X GET "http://localhost:8001/predict/15" \
     -H "Authorization: Bearer YOUR_TOKEN"
     ```

---

## 🛠️ **Recommendations for Improvement**

1. **Enhanced Metrics Tracking**:

   - Add metrics like ROC/AUC, confusion matrix, and precision-recall curves for better model evaluation.
   - Integrate Prometheus and Grafana for real-time performance monitoring.

2. **CI/CD Automation**:

   - Set up GitHub Actions or GitLab CI pipelines for automated testing and deployment.
   - Automate the promotion of models to production after evaluation.

3. **Improved Security**:

   - Implement token expiration and renewal mechanisms.
   - Add rate-limiting and IP whitelisting for APIs.

4. **Unit Testing**:

   - Write unit tests for all components (preprocessing, training, inference).
   - Automate test execution using CI/CD tools.

5. **Documentation Enhancements**:
   - Include step-by-step examples for running the pipeline.
   - Provide detailed usage guides for each service.

---

## 📊 **Conclusion**

This repository demonstrates a well-structured and scalable approach to MLOps. By leveraging tools like Docker, DVC, MLflow, and FastAPI, it ensures reproducibility, modularity, and security throughout the machine learning lifecycle. While the current setup is robust, incorporating additional metrics, CI/CD automation, and enhanced security measures will further solidify its capabilities.

Feel free to contribute, report issues, or suggest improvements! 🌟
