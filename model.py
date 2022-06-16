import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

class Model(nn.Module):
    def __init__(self, model_path, dropout=None, rdrop_coef=0.0, num_labels=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.output_hidden_states = True
        self.ptm = AutoModel.from_pretrained(model_path, config=self.config)
        # dropout 0.1-0.3
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(self.config.hidden_size * 3, self.config.hidden_size * 3),
            nn.Tanh(),
            self.dropout,
            nn.Linear(self.config.hidden_size * 3, num_labels),
        )

        self.regressor = nn.Linear(self.config.hidden_size * 3, num_labels)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = Rdrop_loss
        
        

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                do_evaluate=False,
                pos=None,
                task=1):

        output = self.ptm(input_ids, attention_mask, token_type_ids)
        if self.config.model_type in ['deberta', "electra", 'funnel', 'xlnet']:
            sequence_output, hidden_states = output.last_hidden_state, output.hidden_states
            pooler = sequence_output.mean(dim=1).squeeze()
        else:
            sequence_output, pooler, hidden_states = output[0], output[1], output[2]
        if task == 1:
            seq_out = torch.cat((hidden_states[-1], hidden_states[-2], hidden_states[-3]), dim=1)
        else:
            seq_out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=1)
        keyword_logit = []
        for i in range(input_ids.shape[0]):
            idx = pos[i]
            keyword_logit.append(sequence_output[i][idx])
        keyword_logit = torch.stack(keyword_logit).squeeze()
        seq_avg = torch.mean(seq_out, dim=1)
        #seq_avg = torch.mean(sequence_output, dim=1)
        concat_out = torch.cat((seq_avg, pooler, keyword_logit), dim=1)
        #concat_out = torch.cat((pooler, keyword_logit), dim=-1)
        #logits1 = self.regressor(concat_out)
        logits1 = self.classifier(concat_out)
        

        if self.rdrop_coef > 0 and not do_evaluate:
            output = self.ptm(input_ids, attention_mask, token_type_ids)
            if self.config.model_type in ['deberta', "electra", 'funnel', 'xlnet']:
                sequence_output, hidden_states = output.last_hidden_state, output.hidden_states
                pooler = sequence_output.mean(dim=1).squeeze()
            else:
                sequence_output, pooler, hidden_states = output[0], output[1], output[2]
            seq_out = torch.cat((hidden_states[-1], hidden_states[-2], hidden_states[-3]), dim=1)
            seq_avg = torch.mean(seq_out, dim=1)
            #seq_avg = torch.mean(sequence_output, dim=1)
            concat_out = torch.cat((seq_avg, pooler), dim=1)
            logits2 = self.classifier(concat_out)
            kl_loss = self.rdrop_loss(logits1, logits2)

        else:
            kl_loss = 0.0
        kl_loss = torch.tensor(kl_loss, dtype=torch.float32).cuda()

        return logits1, kl_loss

class SeqCLSModel(nn.Module):
    def __init__(self, model_path, num_labels=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.output_hidden_states = True
        self.config.num_labels = num_labels
        if num_labels == 2:
            self.config.problem_type = "single_label_classification"
        else:
            self.config.problem_type = "multi_label_classification"
    
        self.ptm = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        
       
    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                do_evaluate=False,
                labels=None):

        output = self.ptm(input_ids, attention_mask, token_type_ids, labels=labels)
        loss, logits = output[0], output[1]
        return logits, loss


class AttnModel(nn.Module):
    def __init__(self, model_path, dropout=None, rdrop_coef=0.0, num_labels=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.output_hidden_states = True
        
        self.ptm = AutoModel.from_pretrained(model_path, config=self.config)
        # dropout 0.1-0.3
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        self.attention = nn.Sequential(
            self.dropout,
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.regressor = nn.Linear(self.config.hidden_size, num_labels)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = Rdrop_loss
        
        

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                do_evaluate=False):

        output = self.ptm(input_ids, attention_mask, token_type_ids)
        if self.config.model_type == 'deberta':
            sequence_output, hidden_states = output.last_hidden_state, output.hidden_states
            pooler = sequence_output.mean(dim=1).squeeze()
        else:
            sequence_output, pooler, hidden_states = output[0], output[1], output[2]
        weights = self.attention(sequence_output)
        context_vector = torch.sum(weights * sequence_output, dim=1)
        logits1 = self.regressor(context_vector)
        #logits1 = self.classifier(pooler)
        

        if self.rdrop_coef > 0 and not do_evaluate:
            output = self.ptm(input_ids, attention_mask, token_type_ids)
            if self.config.model_type == 'deberta':
                sequence_output, hidden_states = output.last_hidden_state, output.hidden_states
                pooler = sequence_output.mean(dim=1).squeeze()
            else:
                sequence_output, pooler, hidden_states = output[0], output[1], output[2]
            weights = self.attention(sequence_output)
            context_vector = torch.sum(weights * sequence_output, dim=1)
            logits2 = self.regressor(context_vector)
        else:
            kl_loss = 0.0
        kl_loss = torch.tensor(kl_loss, dtype=torch.float32).cuda()

        return logits1, kl_loss

class CLSModel(nn.Module):
    def __init__(self, model_path, dropout=None, rdrop_coef=0.0, num_labels=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.ptm = AutoModel.from_pretrained(model_path, config=self.config)
        # dropout 0.1-0.3
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = Rdrop_loss

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                do_evaluate=False):

        cls_embedding1 = self.ptm(input_ids, attention_mask, token_type_ids)[1]
        cls_embedding1 = self.dropout(cls_embedding1)
        logits1 = self.classifier(cls_embedding1)

        if self.rdrop_coef > 0 and not do_evaluate:
            cls_embedding2 = self.ptm(input_ids, attention_mask, token_type_ids)[1]
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)

        else:
            kl_loss = 0.0
        kl_loss = torch.tensor(kl_loss, dtype=torch.float32).cuda()

        return logits1, kl_loss

class ModelLastTwoCLS(nn.Module):
    def __init__(self, model_path, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.output_hidden_states = True
        self.ptm = AutoModel.from_pretrained(model_path, config=self.config)
        # dropout 0.1-0.3
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        #self.classifier = nn.Linear(self.config.hidden_size * 3, 2)
        self.classifier = nn.Linear(self.config.hidden_size * 2, 2)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = Rdrop_loss
        
        

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                do_evaluate=False):

        output = self.ptm(input_ids, attention_mask, token_type_ids)
        sequence_output, pooler, hidden_states = output[0], output[1], output[2]
        #cls = torch.cat((pooler, hidden_states[-1][:,0], hidden_states[-2][:,0]), dim=1)
        cls = torch.cat((pooler, hidden_states[-1][:,0], hidden_states[-2][:,0], hidden_states[-3][:,0]), dim=1)
        cls = torch.mean(cls.view(-1, 4, self.config.hidden_size), dim=1)
        
        seq_out = torch.cat((hidden_states[-1], hidden_states[-2], hidden_states[-3]), dim=1)
        seq_avg = torch.mean(seq_out, dim=1)
        concat_out = torch.cat((seq_avg, cls), dim=1)
        cls_embedding1 = self.dropout(concat_out)
        #cls_embedding1 = cls
        logits1 = self.classifier(cls_embedding1)
        
        if self.rdrop_coef > 0 and not do_evaluate:
            output2 = self.ptm(input_ids, attention_mask, token_type_ids)
            sequence_output2, pooler2, hidden_states2 = output2[0], output2[1], output2[2]
            cls2 = torch.cat((pooler2, hidden_states2[-1][:,0], hidden_states2[-2][:,0], hidden_states2[-3][:,0]), dim=1)
            cls2 = torch.mean(cls2.view(-1, 4, self.config.hidden_size), dim=1)
            
            seq_out2 = torch.cat((hidden_states2[-1], hidden_states2[-2], hidden_states2[-3]), dim=1)
            seq_avg2 = torch.mean(seq_out2, dim=1)
            concat_out2 = torch.cat((seq_avg2, cls2), dim=1)
            cls_embedding2 = self.dropout(concat_out2)
            logits2 = self.classifier(cls_embedding2)
            
            kl_loss = self.rdrop_loss(logits1, logits2)

        else:
            kl_loss = 0.0
        kl_loss = torch.tensor(kl_loss, dtype=torch.float32).cuda()

        return logits1, kl_loss

class ModelRCNN(nn.Module):
    def __init__(self, model_path, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.output_hidden_states = True
        self.ptm = AutoModel.from_pretrained(model_path, config=self.config)
        # dropout 0.1-0.3
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        #self.classifier = nn.Linear(self.config.hidden_size * 2, 2)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = Rdrop_loss
        
        # RCNN
        self.bidirectional = True
        self.n_layers = 2
        self.batch_first = True
        self.hidden_dim = 256
        self.drop_out = 0.5
        self.rnn = nn.LSTM(self.config.to_dict()['hidden_size'],
                               self.hidden_dim,
                               num_layers=self.n_layers,
                               bidirectional=self.bidirectional,
                               batch_first=self.batch_first,
                               dropout=self.drop_out)
        self.fc = nn.Linear(self.hidden_dim * self.n_layers, 2)
        

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                do_evaluate=False):

        output = self.ptm(input_ids, attention_mask, token_type_ids)
        sequence_output, pooler, hidden_states = output[0], output[1], output[2]
        sentence_len = sequence_output.shape[1]
        pooler = pooler.unsqueeze(dim=1).repeat(1, sentence_len, 1)
        bert_sentence = sequence_output + pooler
        self.rnn.flatten_parameters()
        output, (hidden, cell) = self.rnn(bert_sentence)
        batch_size, max_seq_len, hidden_dim = output.shape
        out = torch.transpose(output.relu(), 1, 2)
        out = F.max_pool1d(out, max_seq_len).squeeze()
        logits1 = self.fc(out)
        
        if self.rdrop_coef > 0 and not do_evaluate:
            output2 = self.ptm(input_ids, attention_mask, token_type_ids)
            sequence_output2, pooler2, hidden_states2 = output2[0], output2[1], output2[2]
            sentence_len = sequence_output2.shape[1]
            pooler2 = pooler2.unsqueeze(dim=1).repeat(1, sentence_len, 1)
            bert_sentence2 = sequence_output2 + pooler2
            self.rnn.flatten_parameters()
            output2, (hidden, cell) = self.rnn(bert_sentence2)
            batch_size, max_seq_len, hidden_dim = output2.shape
            out2 = torch.transpose(output2.relu(), 1, 2)
            out2 = F.max_pool1d(out2, max_seq_len).squeeze()
            logits2 = self.fc(out2)
            
            kl_loss = self.rdrop_loss(logits1, logits2)

        else:
            kl_loss = 0.0
        kl_loss = torch.tensor(kl_loss, dtype=torch.float32).cuda()

        return logits1, kl_loss


def Rdrop_loss(p, q, pad_mask=None, reduction='none'):
    """
    reduction(obj:`str`, optional): Indicate how to average the loss,
    the candicates are ``'none'`` | ``'batchmean'`` | ``'mean'`` | ``'sum'``.
    If `reduction` is ``'mean'``, the reduced mean loss is returned;
    If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
    if `reduction` is ``'sum'``, the reduced sum loss is returned;
    if `reduction` is ``'none'``, no reduction will be apllied.
    Default is ``'none'``.
    """
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction=reduction)
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction=reduction)

    # pad_mask is used for seq_level tasks
    if pad_mask is not None:
        p_loss = p_loss.masked_fill_(pad_mask, 0.)
        q_loss = q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss
