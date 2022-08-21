from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import pretrained_model_name
import utils
import torch

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name, add_prefix_space=False)


category_value_map_dict = utils.open_object(
    "./artifacts/col_value_to_index_dict.pkl")
catergory_features = list(category_value_map_dict)

numeric_scaler = utils.open_object("artifacts/numeric_scaler.pkl")
numeric_features = list(numeric_scaler.feature_names_in_)


class ViewDataSet(Dataset):

    def __init__(self, df):
        self.df = df
        self.len = len(df)
        
        self.non_text_features_label = numeric_features + \
            catergory_features + ['labels']

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        index = [index]

        data_item = self.df.iloc[index]

        tokenized_inputs = tokenizer(
            text=data_item['next_sri_des'].tolist(),
            text_pair=data_item['hist_sri_des'].tolist(),
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )

        features_dict = data_item[self.non_text_features_label].to_dict("list")
        features_dict.update(tokenized_inputs)
        features_dict = {k: torch.squeeze(torch.tensor(v)) for k, v in features_dict.items()}
        

        return features_dict
