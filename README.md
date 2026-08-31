# SDN-OSHLD v3.0

Optimized Hybrid Learning Framework for QoS Enhancement and Multi-Class Attack Detection in Software-Defined Networks.

## Overview
SDN-OSHLD is a unified machine learning framework for simultaneous network Quality of Service (QoS) prediction and multi-class cyberattack detection in Software-Defined Networks.

This project implements an adaptive ensemble architecture combining regression models for QoS prediction and classifiers for network attack detection. It includes modules for:
- Data processing pipelines (CICIDS2017, NSL-KDD, Synthetic Data)
- Energy-aware routing
- Concept drift detection
- Real-time decision & policy engine
- FastAPI backend inference service
- Web dashboard

## Setup

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Pipeline (To Be Implemented)**
   - Download Data
   - Train Models
   - Start API
   - Start Dashboard
