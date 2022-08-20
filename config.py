# config
from transformers import AutoConfig
import utils

pretrained_model_name = "albert-base-v2"
category_value_map_dict = utils.open_object("./artifacts/col_value_to_index_dict.pkl")
catergory_features = list(category_value_map_dict)

numeric_scaler = utils.open_object("artifacts/numeric_scaler.pkl")
numeric_features = list(numeric_scaler.feature_names_in_)

df_series = utils.open_object("./artifacts/series_table.pkl")
series_features = set(list(df_series.columns))

model_config = AutoConfig.from_pretrained(pretrained_model_name)
model_config.num_lables = 2
model_config.add_pooling_layer = False
model_config.embedding_size = 4

model_config.category_value_map_dict = category_value_map_dict
model_config.series_embedding_size = 16
model_config.target_feature = 'product_series_cms_id' 
model_config.catergory_features = catergory_features
model_config.numeric_features = numeric_features
model_config.series_features = series_features
model_config.bert_output_size = 32
model_config.hidden_sizes = [201,128,64,32]
model_config.dropout = 0.1