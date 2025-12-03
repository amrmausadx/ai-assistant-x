Important Links:

Live website: http://196.218.189.179:5000/

Github Repositry: https://github.com/amrmausadx/ai-assistant-x

Documentation: https://docs.google.com/document/d/1nhBDC9ghk7ZSdY_G5eUqHbKiEZ1oogiCxg994rBTznc/edit?usp=sharing

Presentation: https://www.canva.com/design/DAG6Ge6Oxb0/C4OEM-mx7M9nU-sK7_B5dg/edit

Team members: https://onedrive.live.com/:x:/g/personal/26B83CA047D63BE2/IQCKOdQ5s8eYTKQp0HAZ1vfgASlca8Dsfm2REApJpKN3EWU?resid=26B83CA047D63BE2!s39d4398ac7b34c98a429d07019d6f7e0&ithint=file%2Cxlsx&e=S0o3Qq&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3gvYy8yNkI4M0NBMDQ3RDYzQkUyL0lRQ0tPZFE1czhlWVRLUXAwSEFaMXZmZ0FTbGNhOERzZm0yUkVBcEpwS04zRVdVP2U9UzBvM1Fx

########################################################################
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
