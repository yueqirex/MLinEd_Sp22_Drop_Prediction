import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle
from utils import * 

# dataloader (t-input)
# train
# main
# model

def get_epoch_metrics(model, group_input, group_label, data_loader, mode = None):
    all_pred = []
    all_action_label = []
    for i, batch_uid in enumerate(data_loader):
        if mode == 'train':
            optimizer.zero_grad()
        batch_cid, batch_action, batch_cid_label, batch_action_label = batch_gene_input_data(group_input, group_label, batch_uid, padding_mask=args['padding_mask'])
        output = model(batch_cid, batch_action) #(bs, n_course)
        loss, tar_out_1d, tar_label_1d = model.loss(output, batch_cid_label, batch_action_label, padding_mask = args['padding_mask'])
        if mode == 'train':
            loss.backward()
            optimizer.step()
        
        all_pred.extend(tar_out_1d.data.clone().cpu().numpy().tolist())
        all_action_label.extend(tar_label_1d.data.clone().cpu().numpy().tolist())
        
    all_pred = torch.tensor(all_pred)
    all_action_label = torch.tensor(all_action_label)
    # compute epoch loss
    bce_loss_logits = nn.BCEWithLogitsLoss()
    epoch_loss = bce_loss_logits(all_pred, all_action_label.float()) # (bs, d_model)
    # apply sigmoid to convert to probabilities
    all_pred = torch.sigmoid(all_pred)
    # convert to numpy for auc and acc computation
    all_pred = all_pred.numpy()
    all_binary_pred = (all_pred >= 0.5).astype(np.int8)
    all_action_label = all_action_label.numpy()
    epoch_auc = roc_auc_score(all_action_label, all_pred)
    # print(tar_out.shape)
    # print(tar_label.shape)
    epoch_acc = (all_binary_pred == all_action_label).sum()/(all_action_label.shape[0])
    return epoch_loss, epoch_auc, epoch_acc

def train(model, data, train_loader, val_loader, test_loader):
    best_loss = 1e8
    best_acc = -1
    best_auc = -1
    train_loss_ls = []
    train_auc_ls = []
    train_acc_ls = []
    val_loss_ls = []
    val_auc_ls = []
    val_acc_ls = []

    group_input, group_label = data[0], data[1]
    for epoch in tqdm(range(args['epochs'])):
        train_loss, train_auc, train_acc = get_epoch_metrics(model = model,
                                                            group_input = group_input,
                                                            group_label = group_label,
                                                            data_loader = train_loader,
                                                            mode = 'train')
        val_loss, val_auc, val_acc = get_epoch_metrics(model = model,
                                                        group_input = group_input,
                                                        group_label = group_label,
                                                        data_loader = val_loader,
                                                        mode = 'val')
        print('Epoch {} train_loss - {:4f}, train_auc - {:4f}, train_acc - {:4f}'.format(epoch+1, train_loss, train_auc, train_acc))
        print('Epoch {}   val_loss - {:4f},   val_auc - {:4f},   val_acc - {:4f}'.format(epoch+1, val_loss, val_auc, val_acc))
        # print('\n')
        train_loss_ls.append(train_loss.data.clone().tolist())
        train_auc_ls.append(train_auc)
        train_acc_ls.append(train_acc)
        val_loss_ls.append(val_loss.data.clone().tolist())
        val_auc_ls.append(val_auc)
        val_acc_ls.append(val_acc)
        # record the current best metrics
        if val_loss < best_loss:
            torch.save(model.state_dict(), model_path)
            print('===current best model saved to path===')
            best_loss = val_loss
            best_auc = val_auc
            best_acc = val_acc
        # early stopping
        if (epoch >= 5) and (val_loss_ls[-5:] == sorted(val_loss_ls[-5:])):
            break
    # test for one time
    test_loss, test_auc, test_acc = get_epoch_metrics(model = model,
                                                      group_input = group_input,
                                                      group_label = group_label,
                                                      data_loader = test_loader,
                                                      mode = 'test')
    print('test_loss---{:4f},  test_auc---{:4f},  test_acc---{:4f}'.format(test_loss, test_auc, test_acc))
    return epoch+1,\
        (best_loss, best_auc, best_acc),\
        (test_loss, test_auc, test_acc),\
        (train_loss_ls, train_auc_ls, train_acc_ls, val_loss_ls, val_auc_ls, val_acc_ls)

def eval(model, data, test_loader):
    group_input, group_label = data[0], data[1]
    test_loss, test_auc, test_acc = get_epoch_metrics(model = model,
                                                      group_input = group_input,
                                                      group_label = group_label,
                                                      data_loader = test_loader,
                                                      mode = 'test')
    print('test_loss---{:4f},  test_auc---{:4f},  test_acc---{:4f}'.format(test_loss, test_auc, test_acc))


args = {'n_courses': 7545, # actual num: 3377
        'padding_mask' : 7545,
        'n_actions': 2, # one for padding
        'batch_size': 128,
        'd_model': 256,
        'epochs': 100}

group_input_path = '/home/yueqi/mleddrop/drop_pred/data/group_input.pkl'
group_label_path = '/home/yueqi/mleddrop/drop_pred/data/group_label.pkl'
# for LSTM predictor
# loss_ls_path = '/home/yueqi/mleddrop/drop_pred/data/loss_df.csv'
# test_results_path = '/home/yueqi/mleddrop/drop_pred/data/test_loss_df.csv'
# input_course_index_path = '/home/yueqi/mleddrop/drop_pred/input_course_index_Fall_2020.pkl'
# input_course_emb_path = '/home/yueqi/mleddrop/drop_pred/input_course_emb_Fall_2020.pkl'
# model_path = '/home/yueqi/mleddrop/drop_pred/data/batch_size_{}_d_model_{}.pt'.format(args['batch_size'], args['d_model'])
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# for linear predictor
loss_ls_path = '/home/yueqi/mleddrop/drop_pred/data/linear_loss_df.csv'
test_results_path = '/home/yueqi/mleddrop/drop_pred/data/linear_test_loss_df.csv'
model_path = '/home/yueqi/mleddrop/drop_pred/data/linear_batch_size_{}_d_model_{}.pt'.format(args['batch_size'], args['d_model'])
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

# for the LSTM predictor
if __name__ == '__main__':
    group_input = pd.read_pickle(group_input_path) # stu_id are alingned
    group_label = pd.read_pickle(group_label_path)
    train_uid, val_uid, test_uid = train_val_test_split(group_input)
    # LSTM predictor
    # model = DropPredictor(input_size = args['d_model'],
    #                       hidden_size=args['d_model'],
    #                       num_courses=args['n_courses'],
    #                       num_actions=args['n_actions'],
    #                       device = device)
    # Linear predictor
    global_max_seq_len = get_global_max_seq_len(group_input=group_input)
    model = LinearPredictor(input_size = args['d_model'],
                            hidden_size=args['d_model'],
                            num_courses=args['n_courses'],
                            num_actions=args['n_actions'],
                            global_max_seq_len=global_max_seq_len,
                            padding_mask = args['padding_mask'],
                            device = device)
    model.to(device)
    train_loader = DataLoader(dataset=train_uid, batch_size=128, shuffle=True, num_workers=4, drop_last=False, collate_fn=lambda x: x)
    val_loader = DataLoader(dataset=val_uid, batch_size=128, shuffle=True, num_workers=4, drop_last=False, collate_fn=lambda x: x)
    test_loader = DataLoader(dataset=test_uid, batch_size=128, shuffle=True, num_workers=4, drop_last=False, collate_fn=lambda x: x)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    data = (group_input, group_label)
    epoch,\
    (best_loss, best_auc, best_acc),\
    (test_loss, test_auc, test_acc),\
    (train_loss_ls, train_auc_ls, train_acc_ls, val_loss_ls, val_auc_ls, val_acc_ls) = train(model,
                                                                                            data,
                                                                                            train_loader,
                                                                                            val_loader,
                                                                                            test_loader)

    pd.DataFrame(data = {'epoch': np.arange(len(val_loss_ls)),
                         'train_loss': train_loss_ls,
                         'train_auc': train_auc_ls,
                         'train_acc_ls': train_acc_ls,
                         'val_loss': val_loss_ls,
                         'val_auc_ls': val_auc_ls,
                         'val_acc_ls': val_acc_ls}).to_csv(loss_ls_path)

    pd.DataFrame(data = {'epoch': [epoch],
                         'test_loss': [test_loss],
                         'test_auc': [test_auc],
                         'test_acc': [test_acc]}).to_csv(test_results_path)

    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # eval(model, data, test_loader)

#     with open(input_course_index_path, 'rb') as f:
#         input_course_index = pickle.load(f)
#         input_course_emb = model.get_course_emb(input_course_index)
        
#     with open(input_course_emb_path, 'wb') as f:
#         pickle.dump(input_course_emb, f)