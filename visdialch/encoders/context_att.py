import torch
from torch import nn
from torch.nn import functional as F
from visdialch.utils import GatedTrans


class KVQ_MODULE(nn.Module):
    """docstring for ATT_MODULE"""
    def __init__(self, config, feature_size):
        super(KVQ_MODULE, self).__init__()
        # feature_size: either config["word_embedding_size"], 
        #               or config["img_feature_size"]

        # This could be multiple self-attention
        self.initial_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                feature_size,
                config["lstm_hidden_size"]
            ),
        )
        # Query embedding layer
        self.query_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_hidden_size"],
                config["mlp_kvq_size"]
            ),
        )
        # Key-Value embedding layer
        self.kv_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_hidden_size"],
                config["mlp_kvq_size"]
            ),
        )
        # alpha (weights) embedding layer
        self.alpha = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["lstm_hidden_size"],
                1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, feature):
        # input
        # feature: either `img` or `ques`:
        #   img - shape: (batch_size, num_proposals, img_feature_size)
        #   ques - shape: (batch_size, num_rounds, word_embedding_size)

        batch_size = feature.size(0)
        num = feature.size(1)
        feature_size = feature.size(2)

        feature = feature.view(-1, feature_size) # shape: (batch_size * num, feature_size)
        initial_embed = self.initial_embed(feature) # shape: (batch_size * num, lstm_hidden_size)
        # Query embeddings
        query_embed = self.query_embed(initial_embed) # shape: (batch_size * num, config["mlp_kvq_size"])
        query_embed = query_embed.view(batch_size, num, -1) # shape: (batch_size, num, config["mlp_kvq_size"])
        # Key-Value embeddings
        kv_embed = self.kv_embed(initial_embed) # shape: (batch_size * num, config["mlp_kvq_size"])
        kv_embed = kv_embed.view(batch_size, num, -1) # shape: (batch_size, num, config["mlp_kvq_size"])
        # Alpha mappings
        alpha = self.alpha(initial_embed) # shape: (batch_size * num, 1)
        alpha = alpha.view(batch_size, -1) # shape: (batch_size, num)
        alpha = self.softmax(alpha).unsqueeze(-1) # shape: (batch_size, num, 1), now: sum(alpha[i, :, 0])=1

        return kv_embed, query_embed, alpha


class CONTEXT_ATT_MODULE(nn.Module):
    def __init__(self, config):
        super(CONTEXT_ATT_MODULE, self).__init__()

        self.V_KVQ = KVQ_MODULE(config, config['img_feature_size'])
        self.Q_KVQ = KVQ_MODULE(config, config['word_embedding_size'])

    def forward(self, img, ques):
        """
        Input: 
            img - shape: (batch_size, num_proposals, img_feature_size)
            ques - shape: (batch_size, num_rounds, word_embedding_size)
        """
        
        batch_size = img.size(0)

        # V_kv_embed - shape: (batch_size, num_proposals, config["mlp_kvq_size"])
        # V_query - shape: (batch_size, 1, config["mlp_kvq_size"])
        # V_alpha - shape: (batch_size, num_proposals, 1)

        # Q_kv_embed - shape: (batch_size, num_rounds, config["mlp_kvq_size"])
        # Q_query - shape: (batch_size, 1, config["mlp_kvq_size"])
        # Q_alpha - shape: (batch_size, num_rounds, 1)
        V_kv_embed, V_query, V_alpha = self.V_KVQ(img)
        Q_kv_embed, Q_query, Q_alpha = self.Q_KVQ(ques)

        # V: Query weighted by Alpha
        V_weighted_query_list = V_query * V_alpha # shape: (batch_size, num_proposals, config["mlp_kvq_size"])
        V_weighted_query = torch.sum(V_weighted_query_list, [1]).unsqueeze(1) # shape: (batch_size, 1, config["mlp_kvq_size"])

        # Q: Query weighted by Alpha
        Q_weighted_query_list = Q_query * Q_alpha # shape: (batch_size, num_rounds, config["mlp_kvq_size"])
        Q_weighted_query = torch.sum(Q_weighted_query_list, [1]).unsqueeze(1) # shape: (batch_size, 1, config["mlp_kvq_size"])

        # Attention
        # att = Q_alpha.bmm(V_alpha.transpose(1, 2)) # shape: (batch_size, num_rounds, num_proposals)
        att = Q_weighted_query_list.bmm(V_weighted_query_list.transpose(1, 2)) # shape: (batch_size, num_rounds, num_proposals)
        att = F.normalize(att, dim=-1)

        ### Scores, for pre-training ONLY (start)
        # V final representations 
        V_combine_weights = Q_weighted_query.bmm(V_kv_embed.transpose(1, 2)) # shape: (batch_size, 1, num_proposals)
        V_combine_weights = F.normalize(V_combine_weights, dim=-1) # shape: (batch_size, 1, num_proposals)
        V_final_repr = V_combine_weights.bmm(V_kv_embed).squeeze(1) # shape: (batch_size, config["mlp_kvq_size"])

        # Q final representations 
        Q_combine_weights = V_weighted_query.bmm(Q_kv_embed.transpose(1, 2)) # shape: (batch_size, 1, num_rounds)
        Q_combine_weights = F.normalize(Q_combine_weights, dim=-1) # shape: (batch_size, 1, num_rounds)
        Q_final_repr = Q_combine_weights.bmm(Q_kv_embed).squeeze(1) # shape: (batch_size, config["mlp_kvq_size"])

        self.scores = torch.sum(V_final_repr * Q_final_repr, [1]) # shape: (batch_size, 1)
        ### Scores, for pre-training ONLY (e n d)

        return att