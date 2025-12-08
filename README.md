# End-to-End-Kidney-disease-classification-Deep-learning
## 🩺 Kidney Disease Classification — Model Results & Interpretability

This project builds an end-to-end deep learning system to classify four common kidney conditions from medical images:

- **Cyst**
- **Normal Kidney**
- **Kidney Stone**
- **Tumor**

A custom TensorFlow CNN is trained using a clean MLOps structure (DVC + MLflow + modular pipelines).  
To ensure trust and transparency—especially important in medical AI—we include visual interpretability using **Grad-CAM**.

---

## 🔍 1. Random Prediction Grid (Model Output)

Below is a 3×3 grid of randomly selected images from the validation set.  
Each tile shows the **predicted class** and **model confidence**.

> 📌 *This helps verify the model’s performance across different patient cases.*

![Prediction Grid](outputs/kidney_prediction_grid.png)

---

## 🔥 2. Grad-CAM: Model Explainability for Medical Imaging

Deep learning models are often considered “black boxes.”  
To make the predictions interpretable, we use **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

Grad-CAM highlights **which image regions contributed to the classification**, helping clinicians and ML engineers verify that the model is focusing on anatomically relevant areas.

---

## 🧠 3. 4-Panel Grad-CAM Across All Kidney Classes

Below are Grad-CAM visualizations for:

- Cyst  
- Normal  
- Stone  
- Tumor  

Each row contains:

1. **Original Image**
2. **Heatmap of Activated Regions**
3. **Grad-CAM Overlay**

> 🔎 *This is extremely valuable for medical stakeholders because it reveals whether the model is learning true pathology vs. noise.*

![GradCAM 4 Panel](outputs/gradcam_4panel.png)

---

## 📈 4. Quantitative Performance

| Metric | Score |
|--------|--------|
| **Validation Accuracy** | ~76–78% |
| **Loss Function** | Binary Cross-Entropy + Dice |
| **Optimizer** | Adam |
| **Input Size** | 224×224 |

> ⚕️ *Accuracy is expected to improve using transfer learning (EfficientNet / MobileNetV3), data augmentation, and class-balanced sampling.*

---

## 🧪 5. Training Pipeline (MLOps)

This project follows a **production-ready ML workflow**:

### ✔ Modularized Code (src/components, src/pipeline)  
### ✔ Data Versioning with DVC  
### ✔ Experiment Tracking with MLflow  
### ✔ Docker-ready  
### ✔ AWS-ready (ECR + EC2 deployment pipeline)

This structure enables reproducibility and CI/CD automation.

---

## 🧠 6. Clinical Relevance

Kidney abnormalities such as stones or tumors require early detection.  
This model demonstrates how deep learning can support:

- Radiology workflows  
- Ultrasound/Tomography pre-screening  
- Decision support systems  
- Automated triage

Grad-CAM interpretability ensures the system remains safe and trustworthy.

---

## 📦 7. How to Run

```bash
git clone https://github.com/sohel10/End-to-End-Kidney-disease-classification-Deep-learning
cd End-to-End-Kidney-disease-classification-Deep-learning

conda create -n cnncls python=3.10 -y
conda activate cnncls

pip install -r requirements.txt

python app.py


https://github.com/sohel10/End-to-End-Kidney-disease-classification-Deep-learning
STEP 01- Create a conda environment after opening the repository
conda create -n cnncls python=3.10 -y
conda activate cnncls
STEP 02- install the requirements
pip install -r requirements.txt
# Finally run the following command
python app.py
Now,

open up you local host and port
DVC cmd
dvc init
dvc repro
dvc dag
MLflow
Documentation

cmd

Run this to export as env variables:

1. Login to AWS console.
2. Create IAM user for deployment
#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
3. Create ECR repo to store/save docker image
- Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/
4. Create EC2 machine (Ubuntu)
5. Open EC2 and Install docker in EC2 Machine:
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
6. Configure EC2 as self-hosted runner:
setting>actions>runner>new self hosted runner> choose os> then run command one by one
7. Setup github secrets:
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = simple-app
About MLflow & DVC
MLflow

Its Production Grade
Trace all of your expriements
Logging & taging your model
DVC

Its very lite weight for POC only
lite weight expriements tracker
It can perform Orchestration (Creating Pipelines)