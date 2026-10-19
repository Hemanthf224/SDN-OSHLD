# SDN-OSHLD: Optimized Hybrid Learning Framework for QoS Enhancement and Multi-Class Attack Detection in Software-Defined Networks

## 1. Project Overview

**SDN-OSHLD** is a machine learning-based framework designed to improve Quality of Service (QoS) and detect multiple categories of cyberattacks in Software-Defined Networks (SDNs).

Traditional network management techniques often rely on fixed routing policies and predefined security rules. These approaches may struggle to respond efficiently to changing traffic conditions, network congestion, and evolving cyberattacks.

SDN-OSHLD addresses these challenges by combining machine learning-based QoS prediction, multi-class attack detection, adaptive routing strategies, and a centralized decision-making engine.

The framework is designed to analyze network traffic, predict important network performance metrics, classify potentially malicious traffic, and recommend suitable network management actions.

The system integrates data preprocessing, hybrid machine learning models, concept drift detection, energy-aware routing, a FastAPI inference service, and a web-based monitoring dashboard.

**Primary objective:** Develop a unified framework that supports both network performance optimization and intelligent cybersecurity monitoring within an SDN environment.

> **Implementation status:** This README describes the intended system design. Modules marked as planned or under development must be implemented and tested before the framework can be considered fully operational.

---

## 2. Problem Statement

Software-Defined Networking separates the control plane, which determines network forwarding policies, from the data plane, which forwards packets. This separation enables centralized and programmable network management.

However, SDN environments face several challenges:

- **Network congestion:** Increasing traffic loads can cause delays, packet loss, and reduced throughput.
- **Quality of Service degradation:** Applications such as video streaming, online gaming, and real-time communication require predictable network performance.
- **Cybersecurity threats:** Networks may encounter denial-of-service attacks, scanning activities, brute-force attempts, and other malicious traffic patterns.
- **Changing traffic behavior:** Network traffic distributions can change over time, potentially reducing the accuracy of previously trained models.
- **Resource utilization:** Network routing decisions can affect bandwidth utilization, processing overhead, and energy consumption.
- **Fragmented decision-making:** QoS optimization and attack detection are frequently implemented as separate systems, making coordinated network management more difficult.

SDN-OSHLD aims to address these challenges through a common inference and decision-making pipeline.

---

## 3. Project Objectives

1. Develop a machine learning pipeline for predicting network QoS metrics.
2. Implement multi-class classification of benign and malicious network traffic.
3. Evaluate hybrid ensemble learning strategies for improved predictive performance.
4. Integrate datasets such as CICIDS2017 and NSL-KDD for cybersecurity experiments.
5. Support synthetic network data for controlled testing and simulation.
6. Develop an energy-aware routing strategy that considers network performance and resource utilization.
7. Detect changes in incoming data distributions using concept drift detection.
8. Build a decision and policy engine that combines QoS predictions and security classifications.
9. Expose model inference through a FastAPI backend.
10. Provide a web dashboard for monitoring predictions, detected attacks, and network performance.
11. Evaluate the system using appropriate machine learning and networking metrics.

---

## 4. System Architecture

The framework follows a modular architecture in which network data passes through preprocessing, machine learning inference, and decision-making components.

### High-level workflow

```text
              Network Traffic / Flow Statistics
                            |
                            v
                 Dataset and Data Collection
                            |
                            v
                    Data Preprocessing
                            |
                            v
                  Feature Engineering
                            |
                +-----------+-----------+
                |                       |
                v                       v
          QoS Prediction          Attack Detection
          Regression Models      Classification Models
                |                       |
                v                       v
          Predicted QoS          Attack Class / Score
                |                       |
                +-----------+-----------+
                            |
                            v
                 Concept Drift Monitoring
                            |
                            v
                  Decision and Policy Engine
                            |
                            v
               Routing / Security Recommendations
                            |
                            v
                     FastAPI Backend
                            |
                            v
                     Web Dashboard
```

The architecture is intended to support two major machine learning tasks:

- **Regression:** Predict numerical network performance values.
- **Classification:** Determine whether traffic is benign or belongs to a known attack category.

The final policy engine combines the outputs according to configurable network and security rules.

---

## 5. Core Modules

### 5.1 Data Collection and Dataset Management

The data management module handles importing, validating, and organizing network traffic datasets.

**Supported data sources:**

- **CICIDS2017:** Network traffic data containing benign traffic and several types of attacks.
- **NSL-KDD:** A benchmark intrusion detection dataset containing normal and attack records.
- **Synthetic data:** Artificially generated traffic and network performance measurements for controlled experiments.

The module should support:

- Dataset loading and validation.
- Missing-value analysis.
- Duplicate-record detection.
- Column and data-type validation.
- Label normalization.
- Dataset version tracking.
- Train, validation, and test splitting.

Each dataset must be processed according to its own schema. Dataset-specific attack labels should be mapped to a documented common label taxonomy before combined training is attempted.

**Important:** CICIDS2017 and NSL-KDD contain different feature definitions. Their raw columns should not simply be concatenated. A common feature representation or separate dataset-specific preprocessing pipelines are required.

### 5.2 Data Preprocessing and Feature Engineering

The preprocessing module transforms raw network records into features suitable for machine learning.

Typical operations include:

1. Removing duplicate records where appropriate.
2. Handling missing and invalid values.
3. Replacing infinite numerical values.
4. Encoding categorical variables.
5. Scaling numerical features when required by the selected models.
6. Selecting relevant traffic and performance features.
7. Mapping attack labels into supported classes.
8. Splitting data into training, validation, and testing partitions.

Potential network features include:

- Flow duration.
- Total forward and backward packets.
- Forward and backward bytes.
- Packet length statistics.
- Flow packet rate.
- Flow byte rate.
- Protocol information.
- TCP flag statistics.
- Inter-arrival time statistics.

The exact features depend on the dataset and the available live traffic telemetry.

**Data leakage prevention:** Fit preprocessing transformations only on training data, then apply the fitted transformations to validation and test data. Avoid random splitting strategies that allow nearly identical flows from the same capture or session to appear across different partitions.

### 5.3 Hybrid QoS Prediction Module

The QoS prediction module estimates network performance indicators using regression models.

Possible prediction targets include:

| QoS metric | Description |
|---|---|
| Latency | Time required for traffic to travel through the network |
| Throughput | Amount of data successfully transferred per unit time |
| Packet loss | Proportion of packets that fail to reach their destination |
| Jitter | Variation in packet delay |

The actual targets depend on the availability of ground-truth measurements. These metrics cannot automatically be inferred from an intrusion detection dataset unless suitable labels or measurements exist.

**Proposed model candidates:**

- Random Forest Regressor.
- Gradient Boosting Regressor.
- XGBoost Regressor.
- LightGBM Regressor.
- A weighted ensemble of selected models.

The hybrid model can combine predictions from multiple regressors.

For example:

`Final QoS prediction = w1 × Model 1 prediction + w2 × Model 2 prediction + ...`

where each weight represents the contribution of a model, and the weights sum to one.

The weights should be selected using validation data rather than arbitrary values.

**Output:** Predicted QoS metrics for the supplied network conditions.

**Evaluation metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² where appropriate.

### 5.4 Multi-Class Cyberattack Detection Module

This module classifies network traffic into benign and malicious categories.

The system is designed to support multi-class detection rather than only binary classification.

Possible model candidates include:

- Random Forest Classifier.
- XGBoost Classifier.
- LightGBM Classifier.
- Extra Trees Classifier.
- A soft-voting or stacking ensemble.

The final model selection should be based on validation performance, inference latency, model size, and computational requirements.

**Expected processing pipeline:**

1. Receive network flow features.
2. Validate the feature schema.
3. Apply the saved preprocessing transformations.
4. Generate class probabilities or model scores.
5. Select the predicted class.
6. Return the predicted label and relevant confidence information.
7. Pass the result to the decision engine.

**Potential output classes:** Benign, DoS, DDoS, PortScan, Bot, Brute Force, and other supported attack categories.

The exact class names must match the labels actually present in the selected training datasets. Unsupported classes must not be reported as detected classes without suitable training and validation.

**Evaluation metrics:**

- Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Weighted F1-score.
- Per-class confusion matrix.
- False-positive rate.
- Inference latency.

Macro F1-score is particularly useful when the dataset contains classes with substantially different sample counts.

### 5.5 Energy-Aware Routing Module

The energy-aware routing module is intended to recommend paths based on network performance, resource utilization, and energy-related costs.

Potential routing inputs include:

- Link bandwidth.
- Link utilization.
- Estimated latency.
- Packet loss.
- Queue occupancy.
- Energy cost or a configured energy proxy.
- Security-related link or traffic risk.

A configurable routing objective may be defined as:

\[
J(p)=\alpha D(p)+\beta L(p)+\gamma E(p)
\]

where:

- \(p\) is a candidate network path.
- \(D(p)\) is the estimated path delay.
- \(L(p)\) is a path loss or congestion penalty.
- \(E(p)\) is the estimated energy cost.
- \(\alpha,\beta,\gamma\) are configurable objective weights.

The routing engine selects a feasible path with a low objective value, subject to bandwidth, connectivity, and security constraints.

The terms must be normalized or assigned compatible units before combining them.

The framework may later integrate with an SDN controller such as Ryu or an OpenFlow-compatible controller.

**Safety mechanism:** Routing changes should initially be executed in simulation or dry-run mode. Live flow-rule changes require validation, authorization, rollback support, and protection against routing loops or connectivity loss.

### 5.6 Concept Drift Detection

Concept drift occurs when the statistical relationship between network features and target outcomes changes over time.

For example, a model trained on one traffic distribution may become less reliable when application usage, attack patterns, or network conditions change.

The drift monitoring module can track:

- Changes in feature distributions.
- Changes in model prediction distributions.
- Changes in classification error when verified labels become available.
- Changes in QoS prediction errors.
- Changes in confidence or uncertainty indicators.

Potential techniques include:

- ADWIN.
- Page-Hinkley Test.
- DDM or EDDM for suitable error streams.
- Statistical distribution comparisons.

A drift alert should trigger a monitoring or review workflow. It should not automatically retrain or replace production models without validation.

**Expected output:** Drift score or status, affected features where available, detection timestamp, and a recommendation for further investigation.

### 5.7 Real-Time Decision and Policy Engine

The decision engine combines the results from QoS prediction, attack detection, drift monitoring, and network telemetry.

It converts model outputs into proposed network actions.

Example policy rules:

- If predicted congestion is high, evaluate alternative paths.
- If a high-confidence malicious classification is produced, raise a security alert.
- If a critical attack is detected, recommend an approved mitigation action.
- If concept drift is detected, flag model predictions for review.
- If energy cost is excessive, compare feasible alternative paths while respecting QoS constraints.

The policy engine should use configurable thresholds and explicit action priorities.

Security actions must not rely exclusively on an uncalibrated machine learning confidence score. Policies should include safeguards against false positives and should distinguish between recommendations and executed actions.

**Expected output:** A structured decision containing the relevant predictions, triggered rules, recommended action, and reason for the decision.

### 5.8 FastAPI Backend

FastAPI provides an HTTP interface for accessing the trained models and decision engine.

Proposed endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Check whether the service is responding |
| `GET /health` | Return service health information |
| `POST /predict/qos` | Predict QoS metrics |
| `POST /predict/attack` | Classify a network flow |
| `POST /predict/combined` | Return QoS and security predictions together |
| `GET /metrics` | Expose selected application metrics |
| `GET /model/info` | Return model version and supported classes |

The endpoints are proposed API contracts and must be implemented before they can be used.

The backend should include request validation, structured error handling, model loading, logging, and version information.

### 5.9 Web Dashboard

The dashboard provides a visual interface for observing network conditions and model outputs.

Planned dashboard features include:

- Overall network status.
- Predicted latency and throughput.
- Packet-loss and jitter indicators where available.
- Benign versus malicious traffic summaries.
- Attack-class distribution charts.
- Recent security alerts.
- Drift detection status.
- Routing recommendations.
- Model performance summaries.
- API and inference health status.

The dashboard can be implemented using React, HTML/CSS/JavaScript, or another suitable frontend framework.

It should obtain its data from the backend API instead of loading model files or accessing the training dataset directly.

Real-time updates require a data source and an implemented mechanism such as periodic polling, server-sent events, or WebSockets.

---

## 6. Technology Stack

The following is a proposed technology stack; use only the libraries and frameworks actually installed and implemented in the project.

| Component | Proposed technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Boosting models | XGBoost or LightGBM |
| Drift detection | River or custom statistical detectors |
| Backend API | FastAPI |
| API server | Uvicorn |
| Frontend | React or HTML/CSS/JavaScript |
| Data visualization | Plotly, Chart.js, or equivalent |
| Model persistence | Joblib or an appropriate model-specific format |
| Networking simulation | Mininet |
| SDN controller integration | Ryu or another compatible controller |
| Version control | Git |

The actual stack should be recorded in `requirements.txt` and the project documentation.

---

## 7. Project Directory Structure

A recommended project structure is:

```text
SDN-OSHLD/
│
├── data/
│   ├── raw/
│   │   ├── cicids2017/
│   │   └── nsl_kdd/
│   ├── processed/
│   └── synthetic/
│
├── models/
│   ├── qos/
│   ├── attack_detection/
│   └── preprocessing/
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── validate_data.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── qos/
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── attack_detection/
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── drift/
│   │   └── detector.py
│   │
│   ├── routing/
│   │   └── energy_aware.py
│   │
│   └── decision/
│       └── policy_engine.py
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── services.py
│
├── dashboard/
│   ├── package.json
│   └── src/
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_qos.py
│   ├── test_attack_detection.py
│   └── test_api.py
│
├── configs/
│   └── config.yaml
│
├── requirements.txt
├── .gitignore
└── README.md
```

This is a suggested directory layout. Create only the directories and files needed for the implementation, and update the README to match the actual repository.

---

## 8. Installation and Setup

### Prerequisites

Install the following before starting:

- Python version compatible with the selected ML and API dependencies.
- pip.
- Git, if cloning the repository.
- Node.js and npm if using a React dashboard.
- Mininet and a compatible SDN controller if performing network simulation.

Check the installed versions:

```bash
python --version
python -m pip --version
git --version
node --version
npm --version
```

The Node.js commands are necessary only when using a Node-based frontend.

### Step 1: Clone the Repository

If the project is hosted in a Git repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd SDN-OSHLD
```

Replace the placeholder with the actual repository URL.

If the project already exists on your computer, open a terminal in its root directory instead.

### Step 2: Create a Virtual Environment

On Windows using Command Prompt:

```bat
python -m venv venv
venv\Scripts\activate.bat
```

On Windows using PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

A virtual environment isolates the project's Python dependencies from other Python applications.

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

If `requirements.txt` is already available:

```bash
pip install -r requirements.txt
```

If the dependency file has not yet been created, first install the packages required by the implemented modules.

For example, a basic Python ML and API environment may use:

```bash
pip install pandas numpy scikit-learn fastapi "uvicorn[standard]" joblib
```

Install XGBoost, LightGBM, or River only if the selected implementation requires them.

After validating the environment, record the project dependencies in `requirements.txt`.

---

## 9. Dataset Preparation

### CICIDS2017

Obtain CICIDS2017 from its official Canadian Institute for Cybersecurity dataset resource.

Place the extracted files under:

```text
data/raw/cicids2017/
```

The preprocessing pipeline should verify the available columns, label values, data types, and file encoding before training.

### NSL-KDD

Obtain NSL-KDD from a documented dataset source and place the files under:

```text
data/raw/nsl_kdd/
```

Because its feature schema differs from CICIDS2017, use an appropriate dataset-specific preprocessing configuration.

### Synthetic Data

Synthetic data can be generated to test the complete pipeline when real network telemetry is unavailable.

A useful synthetic dataset may include:

- Flow-level traffic features.
- Simulated latency and throughput.
- Packet-loss estimates.
- Benign and malicious traffic labels.
- Configurable congestion levels.

Synthetic data must be clearly identified as simulated. High model scores on synthetic data do not establish equivalent performance on real network traffic.

---

## 10. Model Training Workflow

The model training process should follow a reproducible sequence.

### Step 1: Validate the Dataset

Check for missing files, invalid labels, duplicate records, malformed columns, and unexpected data types.

### Step 2: Preprocess the Data

Clean the data, encode labels, select features, and fit the required transformations on the training partition.

### Step 3: Train the QoS Models

Train regression models using network features paired with measured or simulated QoS targets.

### Step 4: Train the Attack Detection Models

Train classifiers using labeled network traffic.

### Step 5: Tune the Models

Select hyperparameters and ensemble weights using validation data.

### Step 6: Evaluate Performance

Evaluate the final selected models on an untouched test partition.

### Step 7: Save the Artifacts

Store the fitted models, preprocessing transformations, class mappings, and model configuration.

Example artifact names:

```text
models/
├── qos/
│   └── qos_model.joblib
├── attack_detection/
│   └── attack_model.joblib
└── preprocessing/
    ├── qos_preprocessor.joblib
    └── attack_preprocessor.joblib
```

These files are expected outputs, not files that necessarily exist yet.

---

## 11. Running the Application

The following commands are templates. The corresponding scripts and endpoints must exist before the commands can run successfully.

### Step 1: Run Data Preprocessing

After implementing the preprocessing script:

```bash
python -m src.data.preprocess
```

Expected output: validated and processed datasets in `data/processed/`.

### Step 2: Train the Models

After implementing the training modules:

```bash
python -m src.qos.train
python -m src.attack_detection.train
```

Expected output: saved regression and classification artifacts in `models/`.

### Step 3: Start the FastAPI Backend

If `api/main.py` defines a FastAPI instance named `app`:

```bash
uvicorn api.main:app --reload
```

The local development server typically runs at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is normally available at:

```text
http://127.0.0.1:8000/docs
```

These addresses are valid only after the application starts successfully.

### Step 4: Start the Dashboard

For a React frontend with a configured `package.json`:

```bash
cd dashboard
npm install
npm run dev
```

Use the development URL printed by the frontend tool. Configure the frontend to call the correct backend address.

### Step 5: Test the Complete Pipeline

Submit a test flow through the API and verify that:

1. The input is validated.
2. The appropriate preprocessing transformation is applied.
3. The QoS model returns numerical predictions when requested.
4. The attack classifier returns a supported class and score.
5. The decision engine produces an interpretable recommendation.
6. The API returns a valid response.
7. The dashboard displays the returned data.

---

## 12. Testing and Evaluation

Testing should cover individual modules as well as the complete system.

### Machine Learning Evaluation

**QoS regression:**
- MAE.
- RMSE.
- R².
- Prediction latency.

**Attack detection:**
- Accuracy.
- Macro F1-score.
- Per-class precision and recall.
- Confusion matrix.
- False-positive rate.
- Inference latency.

### System Evaluation

- API response time.
- Requests processed per second.
- Memory usage.
- Model loading time.
- Drift detection delay.
- Routing decision computation time.
- Network throughput, latency, and packet loss before and after routing changes.

### Comparative Experiments

Compare the proposed framework against suitable baselines, such as:

- Individual regression models versus the QoS ensemble.
- Individual classifiers versus the attack detection ensemble.
- Static routing versus energy-aware routing.
- A fixed model versus drift-aware monitoring.
- Separate QoS and security decisions versus the combined policy engine.

Use the same data partitions, comparable resource settings, and clearly defined metrics when making comparisons.

For network experiments, report the simulation topology, traffic generation process, controller configuration, and evaluation conditions.

---

## 13. Security and Reliability Considerations

- Validate all incoming API requests.
- Keep training data separate from production inference.
- Protect model files from unauthorized modification.
- Restrict access to administrative and mitigation endpoints.
- Avoid exposing sensitive network records in application logs.
- Apply authentication and authorization before enabling remote access.
- Use configurable thresholds and safeguards for attack mitigation.
- Monitor model errors and concept drift.
- Test routing changes in simulation before deploying them to real networks.
- Maintain model versions and rollback capabilities.

---

## 14. Current Development Status

The repository should maintain an accurate status for every module.

| Component | Status to record |
|---|---|
| Dataset loading | Pending implementation or tested |
| Preprocessing pipeline | Pending implementation or tested |
| QoS regression models | Pending implementation or tested |
| Multi-class attack detection | Pending implementation or tested |
| Ensemble optimization | Pending implementation or tested |
| Concept drift detection | Pending implementation or tested |
| Energy-aware routing | Pending implementation or tested |
| Decision and policy engine | Pending implementation or tested |
| FastAPI backend | Pending implementation or tested |
| Web dashboard | Pending implementation or tested |
| End-to-end evaluation | Pending implementation or tested |

Update these entries with the actual status of the repository. Do not mark a module as complete until its implementation and tests have been verified.

---

## 15. Future Enhancements

Potential extensions include:

- Live flow collection from an SDN controller.
- Mininet-based network experiments.
- Automated model versioning.
- Online drift monitoring.
- Validated incremental model updates.
- Explainable attack classification.
- Configurable QoS and energy optimization objectives.
- Role-based dashboard access.
- Containerized deployment.
- Integration with monitoring and alerting systems.

These are possible extensions rather than claims about existing functionality.

---

## 16. Expected Outcome

The intended outcome is a modular framework capable of combining network performance prediction, multi-class intrusion detection, drift monitoring, and policy-based network optimization through a common inference service.

Its effectiveness must be demonstrated through reproducible machine learning experiments, API tests, and network-level evaluation.

**Project summary:** SDN-OSHLD combines predictive network analytics and intelligent security monitoring to support QoS-aware and security-conscious decision-making in Software-Defined Networks.
