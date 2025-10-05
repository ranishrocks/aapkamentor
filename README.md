# Career Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-green.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains the core AI/ML components for a sophisticated career recommendation system. The project takes a user's profile—including their skills, interests, and personality traits—and returns a ranked list of the top 5 most suitable career paths with a normalized confidence score.

The entire system is developed, rigorously tested, and deployed as a production-ready, containerized REST API using FastAPI and Docker. A full technical report detailing the methodology and findings can be found in `report.pdf`.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Methodology and Design Decisions](#methodology-and-design-decisions)
- [Setup & Getting Started](#setup--getting-started)
- [Usage](#usage)
- [Running the Test Suite](#running-the-test-suite)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Project Overview

The goal of this project was to build the complete machine learning pipeline for a career recommender, from data engineering to a deployed API. The system frames the problem as a **multi-label classification task**, where a single user can be a good fit for multiple careers.

The final product is a REST API that accepts a JSON object describing a user and returns a ranked list of career recommendations, powered by a calibrated Neural Network and a layer of rule-based heuristics.

---

## Key Features

- **Advanced Feature Engineering**: Raw user data is transformed into 31 meaningful features, including engineered "skill clusters" that capture synergistic skill sets.
- **Rigorous Model Selection**: Random Forest, XGBoost, and a Neural Network were evaluated; the Neural Network was chosen for top-ranking performance.
- **Blended Confidence Score**: Combines model probabilities with rule-based heuristics for explainability and better real-world predictions.
- **Production-Ready API**: High-performance FastAPI application with robust input validation using Pydantic.
- **Automated Documentation**: Interactive Swagger UI available at `/docs`.
- **Comprehensive Testing**: Pytest suite to validate all endpoints and logic.
- **Containerized with Docker**: Fully containerized for reproducible deployment.

---

## Methodology and Design Decisions

1. **Model Selection**: Multi-Layer Perceptron (MLP) Neural Network selected over Random Forest and XGBoost due to better top-k recommendation metrics.
2. **Hyperparameter Optimization**: Baseline architecture already robust; hyperparameter tuning confirmed model stability.
3. **Confidence Score Validation**: Combined heuristics and model output for interpretability without compromising performance.
4. **Error Analysis Insights**: Hybrid user profiles present the main challenge; future improvements will focus on attention-based architectures.

---

## Setup & Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ranishrocks/aapkamentor.git
cd aapkamentor
```

### 2. Create Python Virtual Environment
```bash
python -m venv venv_model
```

Activate the environment:

- **Windows (PowerShell)**:
```powershell
.\venv_model\Scripts\Activate.ps1
```
- **Windows (CMD)**:
```cmd
.\venv_model\Scripts\activate.bat
```
- **Linux / macOS**:
```bash
source venv_model/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Start the API Server
```bash
uvicorn main:app --reload
```
- Access at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Example Python Request
```python
import requests

data = {
    "skills": ["Python", "SQL"],
    "interests": ["Technology"],
    "personality": {"analytical": 0.9, "creative": 0.5, "social": 0.6},
    "education": "Master",
    "experience": 3
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)
print(response.json())
```

### Sample Response
```json
{
  "careers": [
    {"title": "Business Analyst", "confidence": 24},
    {"title": "Research Scientist", "confidence": 20},
    {"title": "Financial Analyst", "confidence": 19},
    {"title": "Software Engineer", "confidence": 19},
    {"title": "Product Manager", "confidence": 18}
  ],
  "model_version": "1.0"
}
```

---

## Running the Test Suite
```bash
pytest
```
- Includes endpoint tests, input validation, and confidence scoring logic.

---

## Docker Deployment

### Build Image
```bash
docker build -t aapkamentor .
```

### Run Container
```bash
docker run -p 8000:8000 aapkamentor
```
- API will be accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Project Structure

```
aapkamentor/
├── career_recommendation_model.pkl   # Serialized trained model
├── model_features.pkl                # Feature names expected by the model
├── main.py                           # FastAPI app
├── maintest2.py                        # Pytest suite for API
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container instructions
├── aapkamentor_report.pdf                         # LaTeX report
├── README.txt                          # This README
└── images/                            # Screenshots and figures
```

---

## Future Improvements

- Support hybrid user profiles using attention-based neural networks.
- Multi-language input support.
- Real-time monitoring & analytics dashboard.
- Expanded personality and education features for improved predictions.
- Web-based dashboard for visualization of recommendations.

---

## License

MIT License
