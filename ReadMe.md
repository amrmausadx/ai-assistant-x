##Conda lines to run

conda create -n mlflow_ai_env
conda init
conda activate mlflow_ai_env
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install bitsandbytes==0.43.3

pip install -r requirements.txt
python -m spacy download en_core_web_sm

mlflow ui

python app.py

#############################################
git add .
git commit -m "adjust read static directory"
git push origin main

###########################################
docker build -t ai-assistant .
##############################################
Important Links:
website link: http://196.218.189.179:5000/
Documentation: https://docs.google.com/document/d/1nhBDC9ghk7ZSdY_G5eUqHbKiEZ1oogiCxg994rBTznc/edit?usp=sharing
Presentation: https://www.canva.com/design/DAG6Ge6Oxb0/C4OEM-mx7M9nU-sK7_B5dg/edit?utm_content=DAG6Ge6Oxb0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
Team members: https://onedrive.live.com/personal/26B83CA047D63BE2/_layouts/15/Doc.aspx?sourcedoc=%7B39d4398a-c7b3-4c98-a429-d07019d6f7e0%7D&action=default&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3gvYy8yNkI4M0NBMDQ3RDYzQkUyL0lRQ0tPZFE1czhlWVRLUXAwSEFaMXZmZ0FTbGNhOERzZm0yUkVBcEpwS04zRVdVP2U9UzBvM1Fx&slrid=49a9dea1-9024-8000-e684-3e4d9dfcd717&originalPath=aHR0cHM6Ly8xZHJ2Lm1zL3gvYy8yNkI4M0NBMDQ3RDYzQkUyL0lRQ0tPZFE1czhlWVRLUXAwSEFaMXZmZ0FTbGNhOERzZm0yUkVBcEpwS04zRVdVP3J0aW1lPUIybXhTNXN3M2tn&CID=5c9c30e3-fc86-4922-a7e2-84367482c929&_SRM=0:G:55


