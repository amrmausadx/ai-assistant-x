##Conda lines to run

conda create -n mlflow_ai_env
conda init
conda activate mlflow_ai_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m spacy download en_core_web_sm

mlflow ui

python app.py

#############################################
git add .
git commit -m ""
git push origin main

###########################################
docker build -t ai-assistant .
