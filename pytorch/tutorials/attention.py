import torch
import torch.nn.functional as F
import torch.nn as nn

# Source: https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention

torch.manual_seed(423)

def sentence_vocab_index(sentence):
    dc = {s:i for i,s 
        in enumerate(sorted(sentence.replace(',', '').split()))}

    return dc

def sentence_vocab_vector(sentence):
    dc = sentence_vocab_index(sentence)
    sentence_int_vec = torch.tensor(
        [dc[s] for s in sentence.replace(',', '').split()]
    )
    # print(sentence_int_vec)
    
    return sentence_int_vec


#
    
def vocab_embedding_model(vocab_size=50000, embedding_size=3):
    embed = torch.nn.Embedding(vocab_size, embedding_size)

    return embed

def embed_sentence(embed_model, sentence):
    if sentence is None  or len(sentence) == 0:
        print("Invalid sentence. Returning None.")
        return None 

    sentence_int_vec = sentence_vocab_vector(sentence)
    emb_sentence = embed_model(sentence_int_vec).detach()

    print(emb_sentence)
    print(emb_sentence.shape)

    return emb_sentence

def attention_matrices(d_emb, query_key_dim=2, value_dim=4):
    d_q = query_key_dim
    d_k = query_key_dim
    d_v = value_dim

    W_query = torch.nn.Parameter(torch.rand(d_emb, d_q))
    W_key = torch.nn.Parameter(torch.rand(d_emb, d_k))
    W_value = torch.nn.Parameter(torch.rand(d_emb, d_v))

    return W_query, W_key, W_value

def get_keys_values(emb_sent, W_key, W_value):
    
    # vector-Matrix product (for each row / sentence token, get weighted sum of key and value columns
    keys = emb_sent @ W_key
    values = emb_sent @ W_value

    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)

    return keys, values


class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
    
    def forward(self, x, masked=False):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T  # unnormalized attention weights
        
        if masked:
            mask = torch.triu(torch.ones(x.shape[0], x.shape[0]), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
            # print(attn_scores)

        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1
        )
        
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v) 
             for _ in range(num_heads)]
        )
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class CrossAttention(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
    
    def forward(self, x_1, x_2, masked=False):           # x_2 is new
        queries_1 = x_1 @ self.W_query
        
        keys_2 = x_2 @ self.W_key          # new
        values_2 = x_2 @ self.W_value      # new
        
        attn_scores = queries_1 @ keys_2.T # new 

        if masked:
            mask = torch.triu(torch.ones(queries_1.shape[0], keys_2.shape[0]), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
            # print(attn_scores)

        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1)
        
        context_vec = attn_weights @ values_2
        return context_vec


def main():
    vocab_size = 50000
    embedding_size = 3

    embed_model = vocab_embedding_model(vocab_size=vocab_size, embedding_size=3)
    
    emb_sent = embed_sentence(embed_model, sentence = 'Life is short, eat dessert first')

    # W_query, W_key, W_value = attention_matrices(emb_sent.shape[1], query_key_dim=2, value_dim=4)

    # keys, values = get_keys_values(emb_sent, W_key, W_value)

    # # Example of second token as query (context is all other tokens (keys) currently, no sliding window...)
    # # Unnormalized attention weights for second token as query
    # omega_2 = W_query[1] @ keys.T
    # print(omega_2)

    # # Normalized attention weights for second token as query, weighted by sqrt(d_k)
    # d_k = keys.shape[1]
    # attention_weights_2 = F.softmax(omega_2 / d_k**0.5, dim=0)
    # print(attention_weights_2)

    # # Context vector for 2nd token (weighted sum of value vectors according to attention weights)
    # context_vec_2 = attention_weights_2 @ values
    # print(context_vec_2.shape)
    # print(context_vec_2)



    # reduce d_out_v from 4 to 1, because we have 4 heads
    d_in, d_out_kq, d_out_v = 3, 2, 4

    # sa = SelfAttention(d_in, d_out_kq, d_out_v)
    # print(sa(emb_sent))

    
    #


    # mha = MultiHeadAttentionWrapper(
    #     d_in, d_out_kq, d_out_v, num_heads=4
    # )

    # # Concats element-wise (e.g. d_out_v * num_heads for each input token (context_vecs.shape[0]))
    # context_vecs = mha(emb_sent)
    
    # print(context_vecs)
    # print("context_vecs.shape:", context_vecs.shape)


    #
    

    crossattn = CrossAttention(d_in, d_out_kq, d_out_v)

    first_input = emb_sent
    second_input = torch.rand(8, d_in)
    
    print("First input shape:", first_input.shape)
    print("Second input shape:", second_input.shape)

    context_vectors = crossattn(first_input, second_input, masked=True)
    
    print(context_vectors)
    print("Output shape:", context_vectors.shape)













if __name__ == "__main__":
    main()

