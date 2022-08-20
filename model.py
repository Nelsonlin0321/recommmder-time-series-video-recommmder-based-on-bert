import torch
from torch import nn
from transformers import AutoModel

class MLP(nn.Module):
    def __init__(self,hidden_sizes,dropout = 0.1) -> None:
        super().__init__()
        
        mlp_list = []
        for i in range(len(hidden_sizes)-1):
            mlp_list.append(nn.Linear(in_features=hidden_sizes[i],out_features=hidden_sizes[i+1],bias=True))
            mlp_list.append(nn.LeakyReLU())
            mlp_list.append(nn.Dropout(p=dropout))
            
        self.mlp = nn.Sequential(*mlp_list)
    
    def forward(self,x):
        return self.mlp(x)


class VedioRecommender(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_config  = model_config
        
        # bert
        self.bert = AutoModel.from_config(model_config)
        self.bert_linear = nn.Linear(self.model_config.hidden_size,self.model_config.bert_output_size, bias=False)
        
        # category embedding
        self.feature_embedding_dict = nn.ModuleDict()
        for feature in self.model_config.catergory_features:
            if feature in self.model_config.series_features:
                category_embeddings = nn.Embedding(len(self.model_config.category_value_map_dict[feature]), self.model_config.series_embedding_size)
            else:
                category_embeddings = nn.Embedding(len(self.model_config.category_value_map_dict[feature]), self.model_config.embedding_size)
                
            category_embeddings.weight.data.uniform_(-0.5,-0.5)
            self.feature_embedding_dict[feature] =category_embeddings
            
        # mlp
        self.mlp = MLP(self.model_config.hidden_sizes,dropout=self.model_config.dropout)
        
        self.ranker = nn.Linear(in_features = self.model_config.hidden_sizes[-1],
                                out_features  = 1,bias=True)

    def mean_pool_concat_embedding(self,embeddings_value):
        embeddings_hist_value = embeddings_value[:,:-1,:]
        embeddings_next_value = embeddings_hist_value[:,-1,:]
        embeddings_hist_mean_value = torch.mean(embeddings_hist_value,dim = 1)
        embeddings_output = torch.concat(
            (embeddings_hist_mean_value,embeddings_next_value),dim=1) 

        return embeddings_output
        
    def forward(self,inputs):
        
        # bert
        bert_encode = self.bert(input_ids=inputs['input_ids'],
                                token_type_ids=inputs['token_type_ids'],
                                attention_mask=inputs['attention_mask'])
        bert_encode = bert_encode.last_hidden_state[:, 0]
        bert_encode = self.bert_linear(bert_encode)
        

        # embedding
        embedding_tensors_list = []
        for feature in self.model_config.catergory_features:
            embedding_ids = inputs[feature]
            
            embedding_tensors = self.feature_embedding_dict[feature](embedding_ids)
            if feature in self.model_config.series_features:
                embedding_tensors = self.mean_pool_concat_embedding(embedding_tensors)
            else:
                embedding_tensors = torch.mean(embedding_tensors,dim = 1)
                
            embedding_tensors_list.append(embedding_tensors)

        embedding_encode = torch.concat(embedding_tensors_list,dim=1)
        
        #numeric
        numeric_tensors_list = []
        for feature in self.model_config.numeric_features:
            tensors  = inputs[feature].view(-1,1)
            numeric_tensors_list.append(tensors)

        numeric_encode = torch.concat(numeric_tensors_list,dim=1)
        
        all_features_encode = torch.concat(
            [bert_encode,embedding_encode,numeric_encode],
            dim=1)
        
        all_features_encode = self.mlp(all_features_encode)
        
        scores = self.ranker(all_features_encode)
        
        scores = torch.sigmoid(scores)
        
        return scores
        

        
                
