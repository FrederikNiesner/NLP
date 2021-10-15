import kaggle.api
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Download all files of a dataset
# Signature: dataset_download_files(dataset, path=None, force=False, quiet=True, unzip=False)
# api.dataset_download_files('avenn98/world-of-warcraft-demographics')

# downoad single file
# Signature: dataset_download_file(dataset, file_name, path=None, force=False, quiet=True)

api.dataset_download_files('/shashikant9198/nlp-and-glove-word-embeddings-sentimental-analysis', path='/Users/fred/OneDrive - Adobe/Data/NLP_sentiment/Kaggle_Files')