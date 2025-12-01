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
Live website: http://196.218.189.179:5000/
Github Repositry: https://github.com/amrmausadx/ai-assistant-x
Documentation: https://docs.google.com/document/d/1nhBDC9ghk7ZSdY_G5eUqHbKiEZ1oogiCxg994rBTznc/edit?usp=sharing
Presentation: https://www.canva.com/design/DAG6PWGAW30/P0FaBERt-VJD2NryMQ6EyA/edit
