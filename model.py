from transformers import AlbertPreTrainedModel,AlbertConfig,AlbertModel
from torch import nn
from typing import Optional,Union,Tuple
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import MSELoss,CrossEntropyLoss,BCEWithLogitsLoss

from transformers import AutoConfig
import utils

pretrained_model_name = "albert-base-v2"
category_value_map_dict = utils.open_object("./artifacts/col_value_to_index_dict.pkl")
catergory_features = list(category_value_map_dict)

numeric_scaler = utils.open_object("artifacts/numeric_scaler.pkl")
numeric_features = list(numeric_scaler.feature_names_in_)

df_series = utils.open_object("./artifacts/series_table.pkl")
series_features = list(df_series.columns)
series_features.remove("sri_des")
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
# model_config.bert_output_size = 64
# model_config.hidden_sizes = [201+32,128,32]
model_config.dropout = 0.1



class MLP(nn.Module):
    def __init__(self, hidden_sizes, dropout=0.1) -> None:
        super().__init__()

        self.mlp = nn.Sequential()
        for i in range(len(hidden_sizes)-1):
            self.mlp.add_module(f'mlp-layers-{i}-Linear', nn.Linear(
                in_features=hidden_sizes[i], out_features=hidden_sizes[i+1], bias=True))
            self.mlp.add_module(f'mlp-layers-{i}-LeakyReLU', nn.LeakyReLU())
            self.mlp.add_module(
                f'mlp-layers-{i}-Dropout', nn.Dropout(p=dropout))

    def forward(self, x):
        return self.mlp(x)
    
class VedioRecommender(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.num_labels = model_config.num_labels
        self.model_config = model_config 
        self.config = config

        # bert
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.model_config.num_labels)
        # Initialize weights and apply final processing
        
        self.post_init()
    
        # category embedding
        self.feature_embedding_dict = nn.ModuleDict()
        for feature in self.model_config.catergory_features:
            if feature in self.model_config.series_features:
                category_embeddings = nn.Embedding(len(
                    self.model_config.category_value_map_dict[feature]), self.model_config.series_embedding_size)
            else:
                category_embeddings = nn.Embedding(len(
                    self.model_config.category_value_map_dict[feature]), self.model_config.embedding_size)

            category_embeddings.weight.data.uniform_(-0.1, -0.1)
            self.feature_embedding_dict[feature] = category_embeddings

        # series
        self.series_feature_size = len(self.model_config.series_features)*self.model_config.series_embedding_size*2
        # self.series_linear = nn.Linear(self.series_feature_size, 1, bias=True)

        # category
        self.category_feature_size = len(
            [f for f in self.model_config.catergory_features if f not in self.model_config.series_features]) * self.model_config.embedding_size
        # self.category_linear = nn.Linear(self.category_feature_size*self.model_config.series_embedding_size, 1, bias=True)

        # numeric
        self.numeric_feature_size = len(self.model_config.numeric_features)
        # self.numeric_linear = nn.Linear(self.numeric_feature_size, 1, bias=True)
        
        
        # # mlp
        # self.mlp = MLP(self.model_config.hidden_sizes,dropout=self.model_config.dropout)
        
        #
        self.all_feature_size = config.hidden_size + self.series_feature_size  + self.category_feature_size  + self.numeric_feature_size
        
        # self.classifier = nn.Linear(config.hidden_size, self.model_config.num_labels)
        
        self.classifier = nn.Linear(in_features=self.all_feature_size,
                                out_features=self.model_config.num_labels, bias=True)

    def mean_pool_concat_embedding(self,embeddings_value):
        embeddings_hist_value = embeddings_value[:,:-1,:]
        embeddings_next_value = embeddings_hist_value[:,-1,:]
        embeddings_hist_mean_value = torch.mean(embeddings_hist_value,dim = 1)
        embeddings_output = torch.concat(
            (embeddings_hist_mean_value,embeddings_next_value),dim=1) 

        return embeddings_output
    
    def forward(
        self,
        inputs:Optional[dict]=None,
    ) -> Union[SequenceClassifierOutput, Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.albert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
        )
        
        bert_encode = outputs[1]
        bert_encode = self.dropout(bert_encode)
        
        labels = inputs['labels']

        # category embedding
        series_embedding_tensors_list = []
        for feature in self.model_config.series_features:
            embedding_ids = inputs[feature]
            embedding_tensors = self.feature_embedding_dict[feature](
                embedding_ids)
            embedding_tensors = self.mean_pool_concat_embedding(
                embedding_tensors)
            series_embedding_tensors_list.append(embedding_tensors)

        series_embedding_encode = torch.concat(
            series_embedding_tensors_list, dim=1)

        
        # category embedding
        category_embedding_tensors_list = []
        for feature in self.model_config.catergory_features:
            if feature not in self.model_config.series_features:
                embedding_ids = inputs[feature]
                embedding_tensors = self.feature_embedding_dict[feature](
                    embedding_ids)
                
                embedding_tensors = torch.mean(embedding_tensors, dim=1)
                category_embedding_tensors_list.append(embedding_tensors)

        category_embedding_encode = torch.concat(category_embedding_tensors_list, dim=1)

        # numeric
        numeric_tensors_list = []
        for feature in self.model_config.numeric_features:
            tensors = inputs[feature].view(-1, 1)
            numeric_tensors_list.append(tensors)

        numeric_encode = torch.concat(numeric_tensors_list, dim=1)

        all_output = torch.concat([bert_encode, series_embedding_encode,category_embedding_encode,numeric_encode],dim=1)
        
        logits = self.classifier(all_output)

        # pooled_output = outputs[1]

        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )