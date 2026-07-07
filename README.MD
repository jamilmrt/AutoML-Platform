# AutoML Platform 🚀

A modern, SaaS-grade Automated Machine Learning platform that allows users to upload raw CSV datasets and instantly train, evaluate, and deploy machine learning models—no coding required.

The platform automatically detects problem types (Classification vs. Regression) based on data types, falls back to Unsupervised Clustering when no target is provided, and generates comprehensive, interactive Exploratory Data Analysis (EDA) reports.

---

## ✨ Key Features

* **Smart Pipeline Routing:** Automatically categorizes tasks into Classification, Regression, or K-Means Clustering based on user input and data types.
* **Automated EDA:** Generates interactive HTML profiling reports and PowerBI-compatible JSON exports using `ydata-profiling`.
* **Asynchronous Processing:** Utilizes Redis and Celery to process large datasets in the background without freezing the UI.
* **Modern SaaS Dashboard:** Built with Next.js and Tailwind CSS, featuring a drag-and-drop dropzone, live progress bars, and dynamic Day/Night mode.
* **One-Click Deployment:** Download the winning Scikit-Learn `.joblib` pipeline (including data imputers and scalers) ready for production inference.

---

## 🏗️ Architecture & Tech Stack

This project uses a decoupled, microservice-style architecture containerized with Docker.

**Frontend:**
* **Framework:** Next.js (React / App Router)
* **Styling:** Tailwind CSS (with dynamic theme switching)
* **File Handling:** `react-dropzone`

**Backend / Machine Learning:**
* **API:** Flask & Flask-CORS
* **Task Queue:** Celery
* **Message Broker:** Redis
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Reporting:** ydata-profiling

**Infrastructure:**
* **Containerization:** Docker & Docker Compose
* **Storage:** Shared Docker volumes for models and HTML/JSON reports

---

## 🚀 Getting Started

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* [Node.js](https://nodejs.org/) (v18+) installed for running the frontend.

### 1. Boot the Backend Infrastructure
The backend relies on three interconnected Docker containers (API, Worker, and Redis).
Open your terminal, navigate to the root directory, and run:
```bash
docker-compose up --build

📂 Project Structure

automl-project/
│
├── backend/
│   ├── app.py              # Flask API entry point
│   ├── tasks.py            # Celery ML worker logic
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend container configuration
│
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   └── AutoMLUploader.tsx  # Main SaaS Dashboard UI
│   │   ├── globals.css     # Tailwind imports
│   │   └── page.tsx        # Next.js root page
│   ├── tailwind.config.ts  # Tailwind styling config
│   └── package.json        # Node dependencies
│
└── docker-compose.yml      # Orchestrates Redis, API, and Worker