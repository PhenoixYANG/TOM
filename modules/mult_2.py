import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder
mult_2_args:{
        'attn_dropout':0.1, 
        'attn_dropout_v':0.1,
        'attn_mask':True, 
        'batch_chunk':1, 
        'batch_size':64, 
        'clip':0.8, 
        'convdim':60, 
        'criterion':'CrossEntropyLoss', 
        'data_path':'data', 
        'dataset':'Eu-senti', 
        'embed_dropout':0.25, 
        'f':'', 
        'ionly':False, 
        'layers':5, 
        'log_interval':30, 
        'loss_weight':[13.392330383480825, 3.736625514403292, 1.520428667113195], 
        'lr':5e-05, 
        'model':'MULT', 
        'model_name':'MULTModel', 
        'n_test':1702, 
        'n_train':13620, 
        'n_valid':1702, 
        'name':'mult', 
        'nlevels':5, 
        'no_cuda':False, 
        'num_epochs':20, 
        'num_heads':5, 
        'num_images':3, 
        'optim':'AdamW', 
        'orig_d_l':0, 
        'orig_d_v':0, 
        'out_dropout':0.1, 
        'output_dim':3, 
        'pre_train':False, 
        'relu_dropout':0.1, 
        'res_dropout':0.1, 
        'seed':1112, 
        'sentis':[1017, 3645, 8958, 0, 0], 
        'sim_level':0.3, 
        'tonly':False, 
        'use_cuda':True, 
        'when':20, 
        'without_sim':False
    }

class mult2args():
    def __init__(self,**kwargs):
        self.attn_dropout=0.1  
        self.attn_dropout_v=0.1  
        self.attn_mask=True  
        self.batch_chunk=1  
        self.batch_size=64 
        self.clip=0.8 
        self.convdim =80 
        self.criterion='CrossEntropyLoss' 
        self.data_path='data' 
        self.dataset='MVSA_M' 
        self.embed_dropout=0.25 
        self.f='' 
        self.ionly=False 
        self.layers=5 
        self.log_interval=30 
        self.loss_weight=[13.392330383480825,3.736625514403292,1.520428667113195] 
        self.lr=5e-05 
        self.model='MULT' 
        self.model_name='MULTModel' 
        self.n_test=1702 
        self.n_train=13620 
        self.n_valid=1702 
        self.name='mult' 
        self.nlevels=5 
        self.no_cuda=False 
        self.num_epochs=20 
        self.num_heads=8 
        self.num_images=3 
        self.optim='AdamW' 
        self.orig_d_l=0 
        self.orig_d_v=0 
        self.out_dropout=0.1 
        self.output_dim=3 
        self.pre_train=False 
        self.relu_dropout=0.1 
        self.res_dropout=0.1 
        self.seed=1112 
        self.sentis=[1017,3645,8958,0,0] 
        self.sim_level=0.3 
        self.tonly=False 
        self.use_cuda=True 
        self.when=20 
        self.without_sim=False

class MyModel(nn.Module):
    def __init__(self, hyp_params, languagemodel, visionmodel) -> None:
        super(MyModel, self).__init__()
        self.language_model = languagemodel
        # self.vision_model=visionmodel
        self.orig_d_l, self.orig_d_v = languagemodel.config.hidden_size, visionmodel.num_features
        self.d_l, self.d_v, self.d_t = hyp_params.convdim, hyp_params.convdim, hyp_params.convdim
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.tonly, self.ionly = hyp_params.tonly, hyp_params.ionly

        #combined_dim = self.d_l  + self.d_v
        combined_dim = self.d_l + self.d_v
        # This is actually not a hyperparameter :-)
        output_dim = hyp_params.output_dim

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(40, self.d_l,
                                kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(1, self.d_v,
                                kernel_size=1, padding=0, bias=False)
        self.proj_t = nn.Conv1d(40, self.d_v,
                                kernel_size=1, padding=0, bias=False)
        # 2. Crossmodal Attentions (only visual and text)
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_t = self.get_network(self_type='vt')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_lv_mem = self.get_network(self_type='mem', layers=3)
        self.trans_vt_mem = self.get_network(self_type='mem', layers=3)
        self.trans_all_mem = self.get_network(self_type='all_mem', layers=3)
        # Projection layers
        # self.channal_fusion=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(1,1))
        self.proj1 = nn.Linear(self.orig_d_l, self.orig_d_l)
        self.proj2 = nn.Linear(self.orig_d_l, self.orig_d_l)
        self.out_layer = nn.Linear(self.orig_d_l, output_dim)

    def forward(self, text, image, topic,text_mask,topic_mask):
        x_l, x_v, x_t = self.use_pretrained(text, image, topic,text_mask,topic_mask)
        return self.mult_theme(x_l, x_v, x_t)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'vl']:
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout
        elif self_type in ['v', 'lv', 'vt']:
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout
        elif self_type == 'mem':
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout
        elif self_type == 'all_mem':
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def use_pretrained(self, text, image, topic,text_mask,topic_mask):
        self.language_model.eval()
        text = text.squeeze(1)
        topic = topic.squeeze(1)
        with torch.no_grad():
            text = self.language_model(input_ids=text,attention_mask=text_mask)['last_hidden_state']#['pooler_output'].unsqueeze(1)
            topic = self.language_model(input_ids=topic,attention_mask=topic_mask)['last_hidden_state']
        return text, image, topic

    def mult_theme(self, x_l, x_v, x_t):
        x_l = F.dropout(x_l,
                        p=self.embed_dropout, training=self.training)
        # Project the textual/visual/audio features by Conv1d
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)  # conv1d
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_t = self.proj_t(x_t)
        proj_x_v = proj_x_v.permute(1, 0, 2)
        proj_x_l = proj_x_l.permute(1, 0, 2)
        proj_x_t = proj_x_t.permute(1, 0, 2)
        # (T) --> V
        h_v_with_ts = self.trans_v_with_t(proj_x_v, proj_x_t, proj_x_t)
        # (V) --> L
        h_l_with_vs = self.trans_l_with_v(proj_x_l, h_v_with_ts, h_v_with_ts)    
        # (L) --> V
        h_v_with_ls = self.trans_v_with_l(h_v_with_ts, proj_x_l, proj_x_l)
        h_l_v = torch.cat((h_l_with_vs, h_v_with_ls), dim=0)
        h_l_v = self.trans_lv_mem(h_l_v)
        if type(h_l_v) == tuple:
            h_l_v = h_l_v[0]
        h_l_v = h_l_v[-1]
        last_hs = h_l_v
        last_hs_proj = self.proj2(F.dropout(
            F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, last_hs

class MULTModel(nn.Module):
    def __init__(self, hyp_params, languagemodel, visionmodel):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.language_model = languagemodel
        # self.vision_model=visionmodel
        self.orig_d_l, self.orig_d_v = languagemodel.config.hidden_size, visionmodel.num_features
        self.d_l, self.d_v, self.d_t = hyp_params.convdim, hyp_params.convdim, hyp_params.convdim
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.tonly, self.ionly = hyp_params.tonly, hyp_params.ionly

        #combined_dim = self.d_l  + self.d_v
        combined_dim = self.d_l + self.d_v
        # This is actually not a hyperparameter :-)
        output_dim = hyp_params.output_dim

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(40, self.d_l,
                                kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(1, self.d_v,
                                kernel_size=1, padding=0, bias=False)
        self.proj_t = nn.Conv1d(40, self.d_v,
                                kernel_size=1, padding=0, bias=False)
        # 2. Crossmodal Attentions (only visual and text)
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_t = self.get_network(self_type='vt')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_lv_mem = self.get_network(self_type='mem', layers=3)
        self.trans_vt_mem = self.get_network(self_type='mem', layers=3)
        self.trans_all_mem = self.get_network(self_type='all_mem', layers=3)
        # Projection layers
        # self.channal_fusion=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(1,1))
        self.proj1 = nn.Linear(self.orig_d_l, self.orig_d_l)
        self.proj2 = nn.Linear(self.orig_d_l, self.orig_d_l)
        self.out_layer = nn.Linear(self.orig_d_l, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'vl']:
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout
        elif self_type in ['v', 'lv', 'vt']:
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout
        elif self_type == 'mem':
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout
        elif self_type == 'all_mem':
            embed_dim, attn_dropout = self.orig_d_l, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, image, topic,text_mask,topic_mask):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l, x_v, x_t = self.use_pretrained(text, image, topic,text_mask,topic_mask)
        #x_l, x_v, x_t=text, image, topic
        x_l = F.dropout(x_l,
                        p=self.embed_dropout, training=self.training)
        # Project the textual/visual/audio features by Conv1d
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)  # conv1d
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(1, 0, 2)
        proj_x_l = proj_x_l.permute(1, 0, 2)
        # (V) --> L
        h_l_with_vs = self.trans_l_with_v(
            proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
        h_ls = h_l_with_vs
        # h_ls=self.trans_l_mem(h_ls)
        # if type(h_ls) == tuple:
        #    h_ls = h_ls[0]
        # last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction
        # (L) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_vs = h_v_with_ls
        #h_vs = self.trans_v_mem(h_vs)
        # if type(h_vs) == tuple:
        #    h_vs = h_vs[0]
        #last_h_v = last_hs = h_vs[-1]
        h_lvs = torch.cat([h_ls, h_vs], dim=0)
        h_lvs = self.trans_lv_mem(h_lvs)
        if type(h_lvs) == tuple:
            h_lvs = h_lvs[0]
        last_hs = h_lvs[-1]

        # A residual block
        # last_hs=last_h_l
        last_hs_proj = self.proj2(F.dropout(
            F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        # print(last_hs_proj.shape)
        output = self.out_layer(last_hs_proj)
        return output, last_hs

    def use_pretrained(self, text, image, topic,text_mask,topic_mask):
        self.language_model.eval()
        text = text.squeeze(1)
        topic = topic.squeeze(1)
        with torch.no_grad():
            text = self.language_model(input_ids=text,attention_mask=text_mask)['last_hidden_state']
        return text, image, topic

