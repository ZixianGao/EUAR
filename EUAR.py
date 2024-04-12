import logging
from modules.transformer import TransformerEncoder
from torch.optim import Adam
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import L1Loss, MSELoss
from transformers.models.bert.configuration_bert import BertConfig
import torch.optim as optim
from itertools import chain
from MoE import MoE_block
from global_configs import TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM, DEVICE
from transformers import BertModel

logger = logging.getLogger(__name__)



def xavier_init(m): 
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x
    

class EUAR_BertModel(nn.Module):
    def __init__(self, multimodal_config, d_l):
        super().__init__()
        model_path='./prebert'
        self.config = BertConfig.from_pretrained(model_path)
        self.encoder=BertModel.from_pretrained(model_path)
  
        self.d_l = d_l
        self.proj_l = nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False) #用base为768 用large为1024


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        # visual,
        # acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        embedding_output = self.encoder(input_ids=input_ids,attention_mask=attention_mask)


        sequence_output = embedding_output.last_hidden_state
    #    print(sequence_output.shape)
  

        outputs = sequence_output.transpose(1, 2)
        outputs = self.proj_l(outputs)
       # outputs1 = torch.mean(outputs, dim=2)
        pooled_output = outputs[:, :, -1]

        return pooled_output


beta   = 1e-3




class EUAR(nn.Module):
    def __init__(self, multimodal_config,num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.d_l = 72
        self.num_heads=4

        self.bert = EUAR_BertModel(multimodal_config, self.d_l)
        self.dropout = nn.Dropout(0.5) 
        self.attn_dropout = 0.5 
        self.proj_a = nn.Conv1d(74, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_v = nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_l = nn.Conv1d(768, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.transa = self.get_network(self_type='l', layers=3)
        self.transv = self.get_network(self_type='l', layers=3)

        self.llr = 1e-5  
        self.lr = 2e-5  
        
        
        self.fusion = fusion(self.d_l)

        

        self.optimizer_l = Adam(self.bert.parameters(), lr=self.llr)
        self.optimizer_all = getattr(optim, 'Adam')(chain(self.transa.parameters(), self.transv.parameters(), self.fusion.parameters(), self.proj_a.parameters(), self.proj_v.parameters()), lr=self.lr)

        self.scheduler_all=optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer_all, "min", patience=2, verbose=True, factor=0.9
    )
        
        self.scheduler_l=optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer_l, "min", patience=2, verbose=True, factor=0.9
    )



        self.mean = nn.AdaptiveAvgPool1d(1)

      







    def get_network(self, self_type='l', layers=5):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout

        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=attn_dropout,
                                  relu_dropout=0.3,
                                  res_dropout= 0.3,
                                  embed_dropout=0.2,
                                  attn_mask= False)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        label_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        output_l = outputs


        acoustic = acoustic.transpose(1, 2)
        visual = visual.transpose(1, 2)

        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)

        acoustic = acoustic.permute(2, 0, 1)
        visual = visual.permute(2, 0, 1)

        outputa = self.transa(acoustic)
        outputv = self.transv(visual)
        output_a = outputa[-1]  # 48 50
        output_v = outputv[-1]
        outputf, loss_u = self.fusion(output_l, output_a, output_v, label_ids)


        loss_fct = L1Loss()

        loss_m = loss_fct(outputf.view(-1,), label_ids.view(-1,))

        loss = loss_u + loss_m*10



        self.optimizer_l.zero_grad()
        self.optimizer_all.zero_grad()
        loss.backward(retain_graph = True)
        self.optimizer_all.step()
        self.optimizer_l.step()


        return outputf,self.scheduler_all,self.scheduler_l



    def test(self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):

        output_l = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,)



        acoustic = acoustic.transpose(1, 2)
        visual = visual.transpose(1, 2)

        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)

        acoustic = acoustic.permute(2, 0, 1)
        visual = visual.permute(2, 0, 1)

        outputa = self.transa(acoustic)
        outputv = self.transv(visual)
        output_a = outputa[-1]  # 48 50
        output_v = outputv[-1]
        outputf = self.fusion.test(output_l, output_a, output_v)


        return outputf







class fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.d_l = dim
        self.d_test=1000 #1024
        self.dr_rate=0.1
        self.classfier=nn.Sequential(LinearLayer(self.d_l*3,self.d_l),
                                     nn.LayerNorm(self.d_l,eps=2e-5),
                                     nn.ReLU(),
                                     LinearLayer(self.d_l,128),
                                     nn.LayerNorm(128,eps=2e-5),
                                     nn.ReLU(),
                                     LinearLayer(128,256),
                                     nn.LayerNorm(256),
                                     nn.ReLU(),
                                     LinearLayer(256,128),
                                     nn.LayerNorm(128,eps=2e-5),
                                     nn.ReLU(),
                                     LinearLayer(128,1))
        # build decoder
        self.decoder = nn.Sequential(LinearLayer(self.d_l,1))
        self.img_MoE= MoE_block(num_tokens=16, dim=self.d_l, heads=8, dim_head=64)
        self.text_MoE= MoE_block(num_tokens=16, dim=self.d_l, heads=8, dim_head=64)
        self.audio_MoE= MoE_block(num_tokens=16, dim=self.d_l, heads=8, dim_head=64)
    def forward(
        self,
        x_l,
        x_a,
        x_v,
        label_ids
    ):
        text_moe_output,text_moe_loss=self.text_MoE(x_l)
        img_moe_output,img_moe_loss=self.img_MoE(x_v)
        audio_output,audio_moe_loss=self.audio_MoE(x_a)
        feature=torch.cat((audio_output,img_moe_output,text_moe_output),dim=1)
        output=self.classfier(feature)    
        loss=audio_moe_loss*1e-5+img_moe_loss*1e-5+text_moe_loss*1e-5
        return output, loss


    def test(
        self,
        x_l,
        x_a,
        x_v
    ):
        text_moe_output,_=self.text_MoE(x_l)
        img_moe_output,_=self.img_MoE(x_v)
        audio_output,_=self.audio_MoE(x_a) 
        feature=torch.cat((audio_output,img_moe_output,text_moe_output),dim=1)
        output=self.classfier(feature)    
        return output