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
git commit -m "add search in preprocessing to load different datasets"
git push origin main

###########################################
docker build -t ai-assistant .
