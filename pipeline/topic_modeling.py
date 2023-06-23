import os
import yaml
import joblib
import logging
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF


logger = logging.getLogger(__name__)

CONFIG_PATH = "./config/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    model_names = config['model_names']



def modeling(vectorized_data, model_name:str, model_save_path:str):
    """
    Trains LSA, LDA, NMF model
        :Args:
            vectorized_data: Vectorized data by Tfidf or Count Vectorizer
            model_name: ['LSA', 'LDA', 'NMF']
            model_save_path: output directory path
    """
    try:
        if model_name.lower() == 'lsa':
            lsa_params = config['lsa_params']
            print(f"Training {model_name.upper()} Model with {lsa_params['n_components']} Topics.")

            lsa_model = TruncatedSVD(**lsa_params)
            lsa_top = lsa_model.fit_transform(vectorized_data)

            # Saving the model
            lsa_model_save_path = os.path.join(model_save_path, model_names['lsa'])
            joblib.dump(lsa_model, lsa_model_save_path)
            print(f"LSA Model saved to [{lsa_model_save_path}]")

            return lsa_model, lsa_top
        elif model_name.lower() == 'lda':
            lda_params = config['lda_params']
            print(f"Training {model_name.upper()} Model with {lda_params['n_components']} Topics.")

            lda_model = LatentDirichletAllocation(**lda_params)
            lda_output = lda_model.fit_transform(vectorized_data)

            # Saving the model
            lda_model_save_path = os.path.join(model_save_path, model_names['lda'])
            joblib.dump(lda_model, lda_model_save_path)
            print(f"LDA Model saved to [{lda_model_save_path}]")

            return lda_model, lda_output
        elif model_name.lower() == 'nmf':
            nmf_params = config['nmf_params']
            print(f"Training {model_name.upper()} Model with {nmf_params['n_components']} Topics.")

            nmf_model = NMF(**nmf_params)
            nmf_output = nmf_model.fit_transform(vectorized_data)

            # Saving the model
            nmf_model_save_path = os.path.join(model_save_path, model_names['nmf'])
            joblib.dump(nmf_model, nmf_model_save_path)
            print(f"NMF Model saved to [{nmf_model_save_path}]")

            return nmf_model, nmf_output
        else:
            raise Exception(f"Wrong model name: [{model_name}] passed.\nAccepted Models are ['LSA', 'LDA', 'NMF']")
    except Exception as e:
        logger.error(f"\nError Occured while Training {model_name.upper()} model \n{e}\n", exc_info=True)


