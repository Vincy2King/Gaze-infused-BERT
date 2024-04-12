import os
import math
import torch
import logging

from torch import nn
# from TorchCR import CRF
# from TorchCRF import CRF
from torch.nn import CrossEntropyLoss, MSELoss

# from transformers.configuration_bert import BertConfig
from transformers import BertConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# from graph_utils import (
#     GraphAttentionNetwork,
#     GATModel,
#     TwoWordPSDProbe,
#     OneWordPSDProbe,
#     L1DistanceLoss,
#     L1DepthLoss,
# )

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
}


def normalization(hs):
    if hs.dim() == 3:
        mean = hs.mean(dim=(0, 1), keepdim=True)
        std = hs.std(dim=(0, 1), keepdim=True)
    elif hs.dim() == 2:
        mean = hs.mean(dim=0, keepdim=True)
        std = hs.std(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unsupported dim: {hs.dim()}")
    return (hs - mean) / (std + 1e-5)


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}

BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        # config.max_position_embeddings=580
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        config.type_vocab_size=2
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
        # print('BertEmbeddings')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # print('input_ids:',input_ids)
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        # print(position_ids.size())
        # exit()
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # print('embeddings 1:',embeddings)
        embeddings = self.LayerNorm(embeddings)
        # print('embeddings 2:', embeddings)
        embeddings = self.dropout(embeddings)
        # print('embeddings 3:', embeddings)
        return embeddings, inputs_embeds

class EyeEmbeddings(nn.Module):
    """Construct the embeddings from eye feature .
    """
    def __init__(self, config):
        super().__init__()
        # if config.revise_gat!='org':
            # print(config.eye_length[config.revise_gat],'-------------------')
        self.eye_embeddings = nn.Embedding(config.eye_length, config.hidden_size, padding_idx=config.pad_token_id)
        self.device = config.device
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self, eye_feature = None
    ):
        # print('EyeEmbeddings')
        # device = eye_feature.device if eye_feature is not None else eye_feature.device
        # print('eye_feature:',eye_feature.shape)
        eye_embeddings = self.eye_embeddings(eye_feature)
        # print(eye_feature)
        # print('------------------')
        # print(eye_embeddings.shape)
        # eye_embeddings = torch.tensor(eye_embeddings, device=self.device)
        # print('eye_embeddings 1:',eye_embeddings.shape,eye_embeddings)
        # if inputs_embeds is None:
        #     inputs_embeds = self.eye_embeddings(eye_feature)
        # print('eye_embeddings:',eye_feature.shape)
        embeddings = eye_embeddings
        inputs_embeds = eye_embeddings
        # print('embeddings:',embeddings.device)
        embeddings = self.LayerNorm(embeddings)
        # print('eye_embeddings 2',embeddings.shape,embeddings)
        embeddings = self.dropout(embeddings)
        # print('eye_embeddings 3',embeddings.shape, embeddings)
        # print('embeddings.deviceL',embeddings.device)
        return embeddings, inputs_embeds

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.scalar_factor = float(config.hidden_size / self.num_attention_heads)

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            addi_key=None,
            addi_value=None,
    ):
        # print('BertSelfAttention')
        bsz, qlen, dim = hidden_states.size()
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
            klen = encoder_hidden_states.size(1)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            klen = qlen

        # bsz, num_heads, len, hidden size
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # O = Attention(Q + GGQl, K + GGKl, V, M, dk),
        if addi_key is not None:
            assert addi_key.dim() == 4
            num_syn_head = addi_key.size(1)
            key_layer[:, :num_syn_head] += addi_key

        if addi_value is not None:
            assert addi_value.dim() == 4
            num_syn_head = addi_value.size(1)
            value_layer[:, :num_syn_head] += addi_value

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores: bsz, nheads, len, len
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * (self.scalar_factor ** -0.5)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,)
        if self.output_attentions:
            outputs = outputs + (attention_probs,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # print('BertSelfOutput')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            addi_key=None,
            addi_value=None,
    ):
        # print('BertAttention')
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states,
            encoder_attention_mask, addi_key=addi_key, addi_value=addi_value
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        # print('BertIntermediate')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # print('BertOutput')
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        # TODO: we do not allow Bert as a Decoder
        assert not self.is_decoder
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            addi_key=None,
            addi_value=None,
    ):
        # print('BertLayer')
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, addi_key=addi_key, addi_value=addi_value
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        # self.revise_gat = config.revise_gat
        # self.revise_edge = config.revise_edge
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        # if config.syntactic_layers == None:
        #     self.syntactic_layers = None
        # else:
        self.syntactic_layers = [int(j) for j in config.syntactic_layers.split(',')]
        self.num_syntax_head = config.num_syntactic_heads
        self.max_syntactic_distance = config.max_syntactic_distance
        '''
        if config.use_syntax:
            gat_out_size = self.head_dim * config.num_gat_head #*2
            # print('gat_out_size:',self.head_dim , config.num_gat_head)
            self.gat_model = GATModel(config)
            # print('output_size:', self.head_dim, self.num_syntax_head)
            output_size = self.head_dim * self.num_syntax_head
            num_syntax_layer = len(self.syntactic_layers)
            assert num_syntax_layer >= 1
            self.graph_attention_k = nn.ModuleList([
                nn.Linear(gat_out_size, output_size) for _ in range(num_syntax_layer)
            ])
            self.graph_attention_v = nn.ModuleList([
                nn.Linear(gat_out_size, output_size) for _ in range(num_syntax_layer)
            ])
        '''
        self.layer = nn.ModuleList([BertLayer(config) for i in range(config.num_hidden_layers)])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_syntax_head, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            word_embeddings=None,
            pos_tag_ids=None,
            dep_tag_ids=None,
            dist_mat=None,
            eye_dist_mat=None,
            FFDs=None,
            GDs=None,
            GPTs=None,
            TRTs=None,
            nFixs=None,
            ffd_embeddings=None
    ):
        # print('BertEncoder')
        all_hidden_states = ()
        all_attentions = ()
        # print(hidden_states)
        # print(hidden_states.size())
        bsz, seq_len, _ = hidden_states.size()

        gat_rep = None
        # print('dist_mat:',dist_mat)
        dist_mat = None
        '''
        if dist_mat is not None:
            assert word_embeddings is not None
            print('word_emb:',word_embeddings.shape)
            if self.revise_gat == 'org':
                org_gat_rep = self.gat_model(word_embeddings, dist_mat, dist_mat, dep_tag_ids, pos_tag_ids)
                # print('dist_mat.shape:',dist_mat.shape)
                # print('gat_rep.shape:',org_gat_rep.shape)
                eye_gat_rep = self.gat_model(word_embeddings, eye_dist_mat, eye_dist_mat, dep_tag_ids, pos_tag_ids, is_threhold=False,is_eye_threhold=True)
                if self.revise_edge == 'fuse':
                    # print('eye_dist_mat:',eye_dist_mat.shape)
                    # eye_gat_rep = self.gat_model(word_embeddings, eye_dist_mat, dep_tag_ids, pos_tag_ids,is_threhold=False)
                    # print('eye_gat_rep:',eye_gat_rep.shape)
                    # gat_rep = torch.cat([gat_rep,eye_gat_rep],dim=2)
                    # new_gat_rep = gat_rep.clone()
                    # new_eye_gat_rep = eye_gat_rep.clone()
                    # total_gat_rep = new_gat_rep + new_eye_gat_rep
                    # print('gat_rep====')
                    gat_rep = org_gat_rep.detach() + eye_gat_rep.detach()#.clone()# + eye_gat_rep
                    # print(gat_rep.shape,gat_rep)
                elif self.revise_edge == 'org':
                    gat_rep = org_gat_rep
                elif self.revise_edge == 'eye':
                    gat_rep = eye_gat_rep
                elif self.revise_edge == 'fuse_dist':
                    # print('=========eye_dist_mat+dist_mat========')
                    # print(eye_dist_mat+dist_mat)
                    gat_rep = self.gat_model(word_embeddings, dist_mat,eye_dist_mat, dep_tag_ids, pos_tag_ids)#eye_gat_rep
                elif self.revise_edge == 'stack':
                    # eye_gat_rep = self.gat_model(word_embeddings, eye_dist_mat,eye_dist_mat, dep_tag_ids, pos_tag_ids)#eye_gat_rep
                    # print('=========stack=========')
                    gat_rep = self.gat_model(org_gat_rep,eye_dist_mat,eye_dist_mat, dep_tag_ids, pos_tag_ids,is_threhold=False,is_text_gat=True).detach()#eye_gat_rep
                    # print('final:',gat_rep.shape)
                elif self.revise_edge == 'stack_vert':
                    gat_rep = self.gat_model(eye_gat_rep, dist_mat, dist_mat, dep_tag_ids, pos_tag_ids,
                                             is_text_gat=True).detach()  # eye_gat_rep

                else:
                    assert 'revise_edge wrong !!!'
            else:
                assert ffd_embeddings is not None
                gat_rep = self.gat_model(word_embeddings + ffd_embeddings, dist_mat, dist_mat, dep_tag_ids, pos_tag_ids)

        # print('gat_rep:',gat_rep.shape,gat_rep)
        # print('FFD:',FFDs.shape,FFDs)
        '''
        gat_layer_idx = 0
        for i, layer_module in enumerate(self.layer):
            # print(i,'here')
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            inputs = {
                'hidden_states': hidden_states,
                'attention_mask': attention_mask,
                'head_mask': head_mask[i],
                'encoder_hidden_states': encoder_hidden_states,
                'encoder_attention_mask': encoder_attention_mask,
            }
            # print('gat:',gat_rep)
            # print(self.syntactic_layers)
            '''
            if gat_rep is not None and i in self.syntactic_layers:
                addi_key = self.graph_attention_k[gat_layer_idx](gat_rep)
                addi_value = self.graph_attention_v[gat_layer_idx](gat_rep)
                print('addi_key:',addi_key.shape)#,addi_key)
                print('addi_value:',addi_value.shape)#,addi_value)
                gat_layer_idx += 1
                inputs["addi_key"] = self.transpose_for_scores(addi_key)
                inputs["addi_value"] = self.transpose_for_scores(addi_value)
                print('inputs["addi_key"] :',inputs["addi_key"])
                exit()
            '''
            addi_key = ffd_embeddings
            addi_value = ffd_embeddings
            # ffd_embeddings = None
            if ffd_embeddings != None:
                inputs["addi_key"] = self.transpose_for_scores(addi_key)
                inputs["addi_value"] = self.transpose_for_scores(addi_value)
            else:
                inputs["addi_key"] = None
                inputs["addi_value"] = None
            # print('addi_key:', type(addi_key),addi_key.shape)  # ,addi_key)
            # print('addi_value:', type(addi_value) ,addi_value.shape)  # ,addi_value)
            # inputs["addi_key"] = self.transpose_for_scores(addi_key)
            # inputs["addi_value"] = self.transpose_for_scores(addi_value)

            layer_outputs = layer_module(**inputs)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        if gat_rep is not None:
            outputs = outputs + (gat_rep,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

    @staticmethod
    def mean_pooling(embeddings, mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        # embeddings is 5d
        input_mask_expanded = mask.unsqueeze(-1).expand(embeddings.size())
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 3)
        sum_mask = torch.clamp(input_mask_expanded.sum(3), min=1e-9)
        return sum_embeddings / sum_mask


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # print('BertPooler')
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    # print('BertPreTrainedModel')
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.
    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.
    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805
    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module
    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:
            (a) For sequence pairs:
                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``
            (b) For single sequences:
                ``tokens:         [CLS] the dog is hairy . [SEP]``
                ``token_type_ids:   0   0   0   0  0     0   0``
            Bert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        **encoder_hidden_states**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``:
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model
            is configured as a decoder.
        **encoder_attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)

# nFix,FFD,GPT,TRT,GD
def calculate_weight(nFix,FFD,GPT,TRT,GD):
    # FFD [w1,w2,w3,...,wn]
    # TRT [w1,w2,w3,...,wn]

    P_nFix,P_FFD,P_GPT,P_TRT,P_GD = [],[],[],[],[]
    for i in range(len(nFix)):
        if sum(nFix)==0:
            P_nFix.append(0)
        else:
            P_nFix.append(nFix[i]/sum(nFix))
        if sum(FFD)==0:
            P_FFD.append(0)
        else:
            P_FFD.append(FFD[i]/sum(FFD))
        if sum(GPT)==0:
            P_GPT.append(0)
        else:
            P_GPT.append(GPT[i]/sum(GPT))
        if sum(TRT)==0:
            P_TRT.append(0)
        else:
            P_TRT.append(TRT[i]/sum(TRT))
        if sum(GD)==0:
            P_GD.append(0)
        else:
            P_GD.append(GD[i]/sum(GD))
    n=len(P_nFix)
    E_nFix, E_FFD, E_GPT, E_TRT, E_GD =0,0,0,0,0
    for i in range(n):
        if P_nFix[i]!=0:
            E_nFix+=P_nFix[i]*math.log(P_nFix[i])

        if P_FFD[i]!=0:
            E_FFD+=P_FFD[i]*math.log(P_FFD[i])

        if P_GPT[i]!=0:
            E_GPT+=P_GPT[i]*math.log(P_GPT[i])

        if P_TRT[i]!=0:
            E_TRT+=P_TRT[i]*math.log(P_TRT[i])

        if P_GD[i]!=0:
            E_GD+=P_GD[i]*math.log(P_GD[i])

    E_nFix*=(-1)*(1/math.log(n))
    E_FFD *= (-1) * (1 / math.log(n))
    E_GPT *= (-1) * (1 / math.log(n))
    E_TRT *= (-1) * (1 / math.log(n))
    E_GD *= (-1) * (1 / math.log(n))

    G_nFix = 1-E_nFix
    G_FFD = 1-E_FFD
    G_GPT = 1-E_GPT
    G_TRT = 1-E_TRT
    G_GD = 1-E_GD

    total = G_nFix+G_FFD+G_GPT+G_TRT+G_GD
    W_nFix = G_nFix/total
    W_FFD = G_FFD/total
    W_GPT = G_GPT/total
    W_TRT = G_TRT/total
    W_GD = G_GD/total

    return [W_nFix,W_FFD,W_GPT,W_TRT,W_GD]

class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # print(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.eye_embeddings = EyeEmbeddings(config)
        self.init_weights()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        # gat_out_size = self.head_dim * config.num_gat_head
        self.num_syntax_head = config.num_syntactic_heads
        output_size = self.head_dim * self.num_syntax_head
        self.eye_embeddings_down = nn.Linear(config.hidden_size, output_size)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            # position_ids=None,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            pos_tag_ids=None,
            dep_tag_ids=None,
            dist_mat=None,
            eye_dist_mat=None,
            FFDs=None,
            GDs=None,
            GPTs=None,
            TRTs=None,
            nFixs=None,
    ):
        # print('BertModel')
        # print('--FFDs--',FFDs)
        """ Forward pass on the Model.
        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.
        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    torch.long
                )  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output, word_embeddings = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        if self.config.revise_gat != 'org':
            # print('ffd.device:',FFDs)
            # 需要在这里算一下各个embedding相加的权重
            # print(FFDs.shape,FFDs)
            # exit()
            batch_size = FFDs.shape[0]
            weight_list =[]
            FFDs_list=FFDs.cpu().numpy().tolist()
            GDs_list = GDs.cpu().numpy().tolist()
            GPTs_list = GPTs.cpu().numpy().tolist()
            nFixs_list = nFixs.cpu().numpy().tolist()
            TRTs_list = TRTs.cpu().numpy().tolist()

            # new_FFDs=[]

            for k in range(batch_size):
                weight = calculate_weight(nFixs_list[k], FFDs_list[k], GPTs_list[k], TRTs_list[k], GDs_list[k])
                weight_list.append(weight)

            # print(weight_list)
            if self.config.revise_gat == 'ffd':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=FFDs)
            elif self.config.revise_gat == 'gd':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=GDs)
            elif self.config.revise_gat == 'gpt':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=GPTs)
            elif self.config.revise_gat == 'trt':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=TRTs)
            elif self.config.revise_gat == 'nfix':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=nFixs)
            elif self.config.revise_gat == 'all':
                ffd_eye_embedding_output, ffd_eye_embeddings = self.eye_embeddings(eye_feature=FFDs)
                gd_eye_embedding_output, gd_eye_embeddings = self.eye_embeddings(eye_feature=GDs)
                # print('gpts:',GPTs)
                gpt_eye_embedding_output, gpt_eye_embeddings = self.eye_embeddings(eye_feature=GPTs)
                trt_eye_embedding_output, trt_eye_embeddings = self.eye_embeddings(eye_feature=TRTs)
                nfix_eye_embedding_output, nfix_eye_embeddings = self.eye_embeddings(eye_feature=nFixs)

                new_ffd_output = ffd_eye_embedding_output.clone()
                new_gd_output = gd_eye_embedding_output.clone()
                new_gpt_output = gpt_eye_embedding_output.clone()
                new_trt_output = trt_eye_embedding_output.clone()
                new_nfix_output = nfix_eye_embedding_output.clone()

                new_ffd = ffd_eye_embeddings.clone()
                new_gd = gd_eye_embeddings.clone()
                new_gpt = gpt_eye_embeddings.clone()
                new_trt = trt_eye_embeddings.clone()
                new_nfix = nfix_eye_embeddings.clone()
                eye_embedding_output = new_ffd_output+new_gd_output+new_gpt_output+new_trt_output+new_nfix_output
                # print('before FFD:',new_ffd)
                #  [W_nFix,W_FFD,W_GPT,W_TRT,W_GD]

                # new_nfix_1 = new_nfix.detach().numpy().tolist()
                # new_ffd_1 = new_ffd.detach().numpy().tolist()
                # new_gpt_1 = new_gpt.detach().numpy().tolist()
                # new_trt_1 = new_trt.detach().numpy().tolist()
                # new_gd_1 = new_gd.detach().numpy().tolist()

                for e in range(new_ffd.shape[0]):
                    # print('before1:',new_ffd[e])
                    new_nfix[e]*=weight_list[e][0]
                    new_ffd[e]*=weight_list[e][1]
                    new_gpt[e]*=weight_list[e][2]
                    new_trt[e]*=weight_list[e][3]
                    new_gd[e]*=weight_list[e][4]
                    # print('after1:', new_ffd[e])
                # new_nfix=torch.tensor(new_nfix_1)
                # new_ffd=torch.tensor(new_ffd_1)
                # new_gpt=torch.tensor(new_gpt_1)
                # new_trt=torch.tensor(new_trt_1)
                # new_gd=torch.tensor(new_gd_1)
                # print('after FFD:',new_ffd)
                eye_embeddings = new_ffd+new_gd+new_gpt+new_trt+new_nfix
                # print('ffd_eye_embedding_loss: ',new_ffd_output.device,new_ffd_output)
                # print('new_ffd:',new_ffd.shape,new_ffd.device)
                # print('eye:',eye_embeddings.device)
                # print('gd_eye_embedding_output:',gd_eye_embedding_output.shape,type(gd_eye_embedding_output))
                # print('gpt_eye_embedding_output:',gpt_eye_embedding_output.shape,type(gpt_eye_embedding_output))
                # print('trt_eye_embedding_output:',trt_eye_embedding_output.shape,type(trt_eye_embedding_output))
                # print('nfix_eye_embedding_output:',nfix_eye_embedding_output.shape,type(nfix_eye_embedding_output))

                # print('eye_embedding_output:',eye_embedding_output.shape,eye_embedding_output)
                # print('eye_embeddings:',eye_embeddings.shape,eye_embeddings)
            elif self.config.revise_gat == 'fuse':
                eye_embedding_output, eye_embeddings = self.eye_embeddings(eye_feature=nFixs)

            # print('+++++++++++++++++++++++')
            # print(embedding_output.shape,word_embeddings.shape)
            # print(eye_embedding_output.shape,eye_embeddings.shape)
            # print('eye_embedding_output:',eye_embedding_output)
            # print('eye_embeddings:',eye_embeddings)

            # eye_embeddings 需要降个维度
            eye_embeddings = self.eye_embeddings_down(eye_embeddings)
            # print('eye_embeddings:',eye_embeddings.shape)
            encoder_outputs = self.encoder(
                embedding_output,#+embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                pos_tag_ids=pos_tag_ids,
                dep_tag_ids=dep_tag_ids,
                dist_mat=dist_mat,
                eye_dist_mat=eye_dist_mat,
                word_embeddings=word_embeddings,
                FFDs=FFDs,
                GDs=GDs,
                GPTs=GPTs,
                TRTs=TRTs,
                nFixs=nFixs,
                ffd_embeddings=eye_embeddings,
            )
        elif self.config.revise_gat == 'org':
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                pos_tag_ids=pos_tag_ids,
                dep_tag_ids=dep_tag_ids,
                dist_mat=dist_mat,
                eye_dist_mat=eye_dist_mat,
                word_embeddings=word_embeddings,
                FFDs=FFDs,
                GDs=GDs,
                GPTs=GPTs,
                TRTs=TRTs,
                nFixs=nFixs,
            )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:
                                                      ]  # add hidden_states and attentions if they are here
        # print(outputs[0][:, 0])
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = BertClassificationHead(config)
        self.cls_normalization = config.batch_normalize
        self.use_structural_loss = config.use_structural_loss
        if self.use_structural_loss:
            head_dim = config.hidden_size // config.num_attention_heads
            model_size = head_dim * config.num_gat_head
            self.dist_probe = TwoWordPSDProbe(model_size, head_dim * 1)
            self.dist_loss = L1DistanceLoss(word_pair_dims=(1, 2))
            self.depth_probe = OneWordPSDProbe(model_size, head_dim * 1)
            self.depth_loss = L1DepthLoss(word_dim=1)
            self.loss_coeff = config.struct_loss_coeff

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos_tag_ids=None,
            dep_tag_ids=None,
            dist_mat=None,
            tree_depths=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            pos_tag_ids=pos_tag_ids,
            dep_tag_ids=dep_tag_ids,
            dist_mat=dist_mat,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        if self.use_structural_loss:
            assert labels is not None
            assert dist_mat is not None
            assert tree_depths is not None
            loss = outputs[0]
            dist_probe_out = self.dist_probe(outputs[-1])
            depth_probe_out = self.depth_probe(outputs[-1])
            struct_loss = self.dist_loss(dist_probe_out,
                                         label_batch=dist_mat,
                                         length_batch=attention_mask.sum(1))[0] + \
                          self.depth_loss(depth_probe_out,
                                          label_batch=tree_depths,
                                          length_batch=attention_mask.sum(1))[0]
            loss = loss + self.loss_coeff * struct_loss
            outputs = (loss,) + outputs[1:]

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with a token classification head on top (a linear layer on top of
                      the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)
class BertForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)
        print('-------1---------')
        print(config)
        print('-----------------')
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_normalization = config.batch_normalize
        self.use_structural_loss = config.use_structural_loss
        if self.use_structural_loss:
            head_dim = config.hidden_size // config.num_attention_heads
            model_size = head_dim * config.num_gat_head
            self.dist_probe = TwoWordPSDProbe(model_size, head_dim * 1)
            self.dist_loss = L1DistanceLoss(word_pair_dims=(1, 2))
            self.depth_probe = OneWordPSDProbe(model_size, head_dim * 1)
            self.depth_loss = L1DepthLoss(word_dim=1)
            self.loss_coeff = config.struct_loss_coeff

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos_tag_ids=None,
            dep_tag_ids=None,
            dist_mat=None,
            eye_dist_mat=None,
            tree_depths=None,
            FFDs =None,
            GDs =None,
            GPTs =None,
            TRTs =None,
            nFixs =None,
    ):
        # print('BertForTokenClassification')
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            pos_tag_ids=pos_tag_ids,
            dep_tag_ids=dep_tag_ids,
            dist_mat=dist_mat,
            eye_dist_mat=eye_dist_mat,
            FFDs=FFDs,
            GDs=GDs,
            GPTs=GPTs,
            TRTs=TRTs,
            nFixs=nFixs,
        )

        sequence_output = outputs[0]
        if self.cls_normalization:
            sequence_output = normalization(sequence_output)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # print('outputs.shape 1:',outputs[2].shape)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # print('outputs.shape 2:', outputs)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # print('loss:',loss)
            outputs = (loss,) + outputs
            # print('output:',len(outputs),outputs)

        if self.use_structural_loss:
            assert labels is not None
            assert dist_mat is not None
            assert tree_depths is not None
            loss = outputs[0]
            dist_probe_out = self.dist_probe(outputs[-1])
            depth_probe_out = self.depth_probe(outputs[-1])
            struct_loss = self.dist_loss(dist_probe_out,
                                         label_batch=dist_mat,
                                         length_batch=attention_mask.sum(1))[0] + \
                          self.depth_loss(depth_probe_out,
                                          label_batch=tree_depths,
                                          length_batch=attention_mask.sum(1))[0]
            loss = loss + self.loss_coeff * struct_loss
            outputs = (loss,) + outputs[1:]

        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(BertClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_normalization = config.batch_normalize

    def forward(self, features, **kwargs):
        # print('BertClassificationHead')
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        if self.cls_normalization:
            x = normalization(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """Bert Model with a token classification head on top (a linear layer on top of
                      the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)
class BertForJointClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForJointClassification, self).__init__(config)

        self.num_labels = config.num_labels
        self.num_slot_labels = config.num_slot_labels

        self.bert = BertModel(config)
        self.intent_classifier = BertClassificationHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.slot_classifier = nn.Linear(config.hidden_size, self.num_slot_labels)
        self.slot_loss_coef = config.slot_loss_coef

        self.cls_normalization = config.batch_normalize
        self.crf = None
        if config.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.use_structural_loss = config.use_structural_loss
        if self.use_structural_loss:
            head_dim = config.hidden_size // config.num_attention_heads
            model_size = head_dim * config.num_gat_head
            self.dist_probe = TwoWordPSDProbe(model_size, head_dim * 1)
            self.dist_loss = L1DistanceLoss(word_pair_dims=(1, 2))
            self.depth_probe = OneWordPSDProbe(model_size, head_dim * 1)
            self.depth_loss = L1DepthLoss(word_dim=1)
            self.loss_coeff = config.struct_loss_coeff

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            sequence_labels=None,
            pos_tag_ids=None,
            dep_tag_ids=None,
            dist_mat=None,
            tree_depths=None,
    ):
        # print('BertForJointClassification')
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            pos_tag_ids=pos_tag_ids,
            dep_tag_ids=dep_tag_ids,
            dist_mat=dist_mat,
        )
        sequence_output = outputs[0]
        intent_logits = self.intent_classifier(sequence_output)

        if self.cls_normalization:
            sequence_output = normalization(sequence_output)
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        outputs = ((intent_logits, slot_logits),) + outputs[2:]
        total_loss = 0
        if labels is not None:
            intent_loss_fct = CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_labels), labels.view(-1))
            total_loss = total_loss + intent_loss

        if sequence_labels is not None:
            if self.crf is not None:
                slot_loss = self.crf(slot_logits, sequence_labels, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = sequence_labels.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), sequence_labels.view(-1))
            total_loss = total_loss + self.slot_loss_coef * slot_loss

        if (labels is not None) or (sequence_labels is not None):
            outputs = (total_loss,) + outputs

        if self.use_structural_loss:
            assert labels is not None
            assert dist_mat is not None
            assert tree_depths is not None
            loss = outputs[0]
            dist_probe_out = self.dist_probe(outputs[-1])
            depth_probe_out = self.depth_probe(outputs[-1])
            struct_loss = self.dist_loss(dist_probe_out,
                                         label_batch=dist_mat,
                                         length_batch=attention_mask.sum(1))[0] + \
                          self.depth_loss(depth_probe_out,
                                          label_batch=tree_depths,
                                          length_batch=attention_mask.sum(1))[0]
            # logger.info('loss: {} + {}'.format(
            #     str(round(loss.item(), 3)), str(round(struct_loss.item(), 3)))
            # )
            loss = loss + self.loss_coeff * struct_loss
            outputs = (loss,) + outputs[1:]

        return outputs  # (loss), logits, (hidden_states), (attentions)
