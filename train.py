import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import transformers
from tqdm import tqdm

import wandb
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data import *
from model import *
from evaluation import *
from utils import *

num_labels = 2


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, epoch, criterion, data_loader, use_wandb=False, task=1, threshold=0.5):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        epoch(obj: 'int'): current epoch
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
    """
    model.eval()
    losses = []
    total_num = 0
    f1s = []
    ps = []
    rs = []
    batch_logits = []

    for batch in data_loader:
        input_ids, token_type_ids, attention_mask, pos, labels = batch['input_ids'], batch['token_type_ids'], batch['attention_mask'], batch['pos'], batch['label']
        total_num += len(labels)
        input_ids, token_type_ids, attention_mask, pos, labels = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda(), pos.cuda(), labels.cuda()
        with torch.no_grad():
            if args.model_type == 'seq_cls':
                logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels.view(-1, num_labels))
                if task == 1:
                    losses.append(loss.item())
                    pred_cal = torch.argsort(logits, dim=-1, descending=True)
                    pred_cal = pred_cal[..., :1]
                    batch_logits.append(pred_cal.cpu().numpy().tolist())
                else:
                    losses.append(loss.item())                    
                    act_fct = nn.Sigmoid()
                    logits = act_fct(logits)
                    pred_cal = (logits > threshold).float()
                    batch_logits.append(pred_cal.cpu().numpy().tolist())
                
            else:
                logits, _ = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, do_evaluate=True, pos=pos, task=task)
                if task == 1:
                    loss = criterion(logits, labels)
                    losses.append(loss.item())

                    pred_cal = torch.argsort(logits, dim=-1, descending=True)
                    pred_cal = pred_cal[..., :1]
                    batch_logits.append(pred_cal.cpu().numpy().tolist())
                    #precision, recall, f1 = metric_cal(labels, pred_cal)
                else:
                    loss = criterion(logits.view(-1, 7), labels.view(-1, 7).float())
                    losses.append(loss.item())
                    
                    act_fct = nn.Sigmoid()
                    logits = act_fct(logits)
                    pred_cal = (logits > threshold).float()
                    batch_logits.append(pred_cal.cpu().numpy().tolist())
                    #precision, recall, f1 = metric_cal(labels, pred_cal, task=2)
                #ps.append(precision)
                #rs.append(recall)
                #f1s.append(f1)
    batch_logits = np.vstack(batch_logits)
    if task == 1:
        labels2file([k for k in batch_logits], '../data/result/dev/task1.txt')
        precision, recall, f1 = eval_task1()
    else:
        labels2file(batch_logits, 'data/result/dev/task2.txt')
        f1 = eval_task2()
    if use_wandb:
        wandb.log({'dev_loss': np.mean(losses), 'dev_f1': f1})
    print('----Validation Results Summary----')
    if task == 1:
        print("epoch: [{}], dev_loss: {:.5}, precision: {:.5}, recall: {:.5}, f1: {:.5}, total_num:{}".format(epoch, np.mean(losses), precision, recall, f1, total_num))
    else:
        print("epoch: [{}], dev_loss: {:.5}, f1: {:.5}, total_num:{}".format(epoch, np.mean(losses), f1, total_num))
    print('-*' * 89)
    model.train()
    return f1


def do_train(args):
    device = torch.device(args.device)
    set_seed(args.seed)

    # dpm = DontPatronizeMe(args.train_set, args.dev_set)
    # dpm.load_task1()
    # dpm.load_task2(return_one_hot=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.ptr_dir)

    if args.task == 1:
        train_set = read_text(args.train_set, task=1, tokenizer=tokenizer, max_len=args.max_seq_length)
        dev_set = read_text(args.dev_set, task=1, tokenizer=tokenizer, max_len=args.max_seq_length)
        print('task1')
    else:
        train_set = read_text(args.train_set, task=2)
        dev_set = read_text(args.dev_set, task=2)
        print('task2')

    train_dataset = DatasetRetriever(train_set)
    dev_dataset = DatasetRetriever(dev_set)

    train_loader = create_dataloader(train_dataset, batch_size=args.train_batch_size, weightsample=args.weightsample)
    dev_loader = create_dataloader(dev_dataset, batch_size=args.eval_batch_size, mode='eval')
    
    if args.task == 1:
        num_labels = 2
    else:
        num_labels = 7
    if args.model_type == 'last2cls':
        print('use last2cls model')
        model = ModelLastTwoCLS(args.ptr_dir, rdrop_coef=args.rdrop_coef, dropout=args.dropout, num_labels=num_labels)
    elif args.model_type == 'attn':
        print('use attn model')
        model = AttnModel(args.ptr_dir, rdrop_coef=args.rdrop_coef, dropout=args.dropout, num_labels=num_labels)
    elif args.model_type == 'clsmodel':
        print('use cls model')
        model = CLSModel(args.ptr_dir, rdrop_coef=args.rdrop_coef, dropout=args.dropout, num_labels=num_labels)
    elif args.model_type == 'seq_cls':
        print('use automodelforsequenceclassification')
        model = SeqCLSModel(args.ptr_dir, num_labels=num_labels)
    else:
        model = Model(args.ptr_dir, rdrop_coef=args.rdrop_coef, dropout=args.dropout, num_labels=num_labels)
    # model = nn.DataParallel(model, device_ids=[0,1])
    model.cuda()
    if args.fgm == 1:
        fgm = FGM(model)

    if args.use_wandb:
        wandb.watch(model)
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        model.load_state_dict(torch.load(args.init_from_ckpt))

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    num_training_steps = math.ceil(len(train_loader) / args.accu_gradient) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    if args.task == 1:
        if args.focal_loss == 1:
            print('use focal loss')
            # focal loss alpha=0.25 gamma=0.75
            if args.focal_alpha > 0:
                criterion = FocalLoss(args.focal_alpha)
            else:
                criterion = FocalLoss()
        else:
            print('use cross entropy loss')
            criterion = nn.CrossEntropyLoss()
    else:
        if args.multilabel_celoss == 1:
            print('use multilabel_categorical_crossentropy loss')
            criterion = MultilabelCELoss()
        else:
            print('use bcewithlogits loss')
            criterion = nn.BCEWithLogitsLoss()
    global_step = 0
    best_f1 = 0.0
    best_epoch = 1
    best_step = 100

    tic_train = time.time()
    for epoch in tqdm(range(1, args.epochs + 1)):
        st_epoch = time.time()
        for step, batch in enumerate(train_loader, start=1):
            input_ids, token_type_ids, attention_mask, pos, labels = batch['input_ids'], batch['token_type_ids'], batch[
                'attention_mask'], batch['pos'], batch['label']
            input_ids, token_type_ids, attention_mask, pos, labels = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda(), pos.cuda(), labels.cuda()
            if args.model_type == 'seq_cls':
                logits, ce_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels.view(-1, num_labels))
                kl_loss = 0
            else:
                logits, kl_loss = model(input_ids=input_ids, attention_mask=attention_mask, task=args.task, pos=pos)
                kl_loss = kl_loss.cpu().numpy()

            if args.task == 1:
                if args.model_type != 'seq_cls':
                    ce_loss = criterion(logits.view(-1, num_labels), labels.view(-1))
                pred_cal = torch.argsort(logits, dim=-1, descending=True)
                pred_cal = pred_cal[..., :1]
                precision, recall, f1 = metric_cal(labels, pred_cal)
            else:
                if args.model_type != 'seq_cls':
                    ce_loss = criterion(logits.view(-1, num_labels), labels.view(-1, num_labels).float())
                pred_cal = (logits > 0.5).int()
                precision, recall, f1 = metric_cal(labels, pred_cal, task=2)
            if kl_loss > 0:
                loss = ce_loss + kl_loss * args.rdrop_coef
            else:
                loss = ce_loss
            loss = loss.mean()
            loss = loss / args.accu_gradient
            global_step += 1
            if global_step % 50 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.4f, ce_loss: %.4f, kl_loss: %.4f, precision: %.4f, recall: %.4f, f1: %.4f, speed: %.2f step/s, time: %.5f"
                    % (global_step, epoch, step, loss, ce_loss, kl_loss, precision, recall, f1,
                       10 / (time.time() - tic_train), (time.time() - tic_train)))
                if args.use_wandb:
                    wandb.log({'loss': loss, 'ce_loss': ce_loss, 'kl_loss': kl_loss, 'f1': f1,
                               'lr': optimizer.param_groups[0]['lr']})
                tic_train = time.time()

            loss.backward()
            # 对抗训练
            if args.fgm == 1:
                fgm.attack()  # embedding被修改了
                optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
                logits_adv, kl_loss_adv = model(input_ids=input_ids, attention_mask=attention_mask, task=args.task)
                kl_loss_adv = kl_loss_adv.cpu().numpy()
                if args.task == 1:
                    if args.model_type != 'seq_cls':
                        ce_loss_adv = criterion(logits_adv.view(-1, num_labels), labels.view(-1))
                else:
                    if args.model_type != 'seq_cls':
                        ce_loss_adv = criterion(logits_adv.view(-1, num_labels), labels.view(-1, num_labels).float())
                loss_adv = ce_loss_adv + kl_loss_adv * args.rdrop_coef
                loss_adv = loss_adv.mean()
                loss_adv.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复Embedding的参数
            # 梯度下降，更新参数
            if step % args.accu_gradient == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            

            if global_step == args.max_steps:
                return
        # evaluate per epoch
        if args.task == 1:
            dev_f1 = evaluate(model, epoch, criterion, dev_loader, use_wandb=args.use_wandb, task=1)
        else:
            dev_f1 = evaluate(model, epoch, criterion, dev_loader, use_wandb=args.use_wandb, task=2, threshold=args.threshold)
        if dev_f1 > best_f1:
            save_dir = os.path.join(args.save_dir, "model_best_step")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_param_path = os.path.join(save_dir, 'model_state.pt')
            torch.save(model.state_dict(), save_param_path)
            best_f1 = dev_f1
            best_epoch = epoch

        print('best epoch: %d, best f1: %.4f' % (best_epoch, best_f1))
        print('-' * 90)
        print('epoch costs %.4f seconds' % (time.time() - st_epoch))
        print('*' * 90)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str, required=True, help="The full path of train_set_file")
    parser.add_argument("--dev_set", type=str, required=True, help="The full path of dev_set_file")
    parser.add_argument("--save_dir", default='./checkpoint', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--ptr_dir", type=str, required=True, help="The full path of pretrain model")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--max_steps', default=-1, type=int,
                        help="If > 0, set total number of training steps to perform.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--eval_step", default=500, type=int, help="Step interval for evaluation.")
    parser.add_argument('--save_step', default=10000, type=int, help="Step interval for saving checkpoint.")
    parser.add_argument("--warmup_ratio", default=0.0, type=float,
                        help="Linear warmup proption over the training process.")
    parser.add_argument("--accu_gradient", type=int, default=1, help="gradient accumulation")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--seed", type=int, default=2021, help="Random seed for initialization.")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default="cuda",
                        help="Select which device to train model, defaults to gpu.")
    # 1 - 5
    parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of"
                                                                      "KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
    parser.add_argument("--dropout", default=0.1, type=float, help='dropout')
    parser.add_argument("--model_type", default='last3hidden', type=str, help='model type')
    parser.add_argument("--use_wandb", default=False, type=bool, help='whether to use wandb')
    parser.add_argument("--model_name", default='ernie-gram', type=str, help='model name')
    parser.add_argument("--task", default=1, type=int, help='task1 is pcl, 2 is multilabel')
    parser.add_argument("--fgm", default=0, type=int, help='adv training')
    parser.add_argument("--focal_loss", default=0, type=int, help='whether to use focal loss')
    parser.add_argument("--focal_alpha", default=0, type=float, help='hyper param of focal loss')
    parser.add_argument("--weightsample", default=0, type=float, help='whether to weight sample')
    parser.add_argument("--kw", default=1, type=float, help='whether to use keyword')
    parser.add_argument("--threshold", default=0.5, type=float, help='hyper param of mlabel threshold')
    parser.add_argument("--multilabel_celoss", default=0, type=int, help='whether to use multilabel_categorical_crossentropy')
    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(name=args.model_name, project='semeval2022_task4', config=args.__dict__)
    do_train(args)
