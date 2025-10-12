##Conda lines to run

conda create -n mlflow_ai_env
conda init
conda activate mlflow_ai_env
pip install -r requirements.txt
python -m spacy download en_core_web_sm

mlflow ui

python app.py



###########################################
docker build -t ai-assistant .
