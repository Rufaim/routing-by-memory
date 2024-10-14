import os
import copy
import enum
import time
import numpy as np
import yaml
import random
import dgl
import argparse
import torch
from dataloader import DataType, load_data
from models import Com_KD_Prob, TeacherModelType, StudentModelType, \
                load_student_model, load_teacher_model
from position_encoding import DeepWalk


class Setting(enum.Enum):
    INDUCTIVE = 'inductive'
    TRANSDUCTIVE = 'transductive'

    def __str__(self):
        return self.value


class CombinedClassificationDistillationLoss(object):
    def __init__(self, lmbda, tau, adv_augmenter=None, adv_weigth=None, feat_distill_weigth=None) -> None:
        
        self.lmbda = float(lmbda)
        self.tau = float(tau)
        self.adv_augmenter = adv_augmenter
        self.adv_weigth = float(adv_weigth)
        self.feat_distill_weigth = float(feat_distill_weigth)
        
        self._main_loss = torch.nn.NLLLoss()
        self._distill_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self._feature_distill_loss = torch.nn.MSELoss()
    
    def classif_loss(self, out, labels):
        loss_main = self._main_loss(out, labels)
        return loss_main * self.lmbda
    
    def distill_loss(self, out, teacher_output):
        out_student = (out/self.tau)
        out_teacher = (teacher_output/self.tau)
        loss_distill = self._distill_loss(out_student, out_teacher)
        return loss_distill * (1 - self.lmbda)
    
    def augment_classif_loss(self, model, feats, labels):
        assert self.adv_augmenter is not None
        assert self.adv_weigth is not None
        
        adv_feats = self.adv_augmenter(model, feats, self._main_loss, labels)
        _, adv_logits, aux_loss = model(adv_feats)
        adv_outputs_labels = adv_logits.log_softmax(dim=1)
        loss_adv_class = self.classif_loss(adv_outputs_labels, labels) + aux_loss
        return self.adv_weigth * loss_adv_class
    
    def augment_distill_loss(self, model, feats, teacher_output):
        assert self.adv_augmenter is not None
        assert self.adv_weigth is not None
        
        adv_feats = self.adv_augmenter(model, feats, self._distill_loss, teacher_output)
        _, adv_logits, aux_loss = model(adv_feats)
        adv_outputs_labels = adv_logits.log_softmax(dim=1)
        loss_adv_class = self.distill_loss(adv_outputs_labels, teacher_output) + aux_loss
        return self.adv_weigth * loss_adv_class
    
    def feature_sim_loss(self, feat_distill_sim, teacher_sim):
        assert self.feat_distill_weigth is not None
        
        out = self._feature_distill_loss(feat_distill_sim, teacher_sim)
        return self.feat_distill_weigth * out 
        


class AdversarialFeatureAugmenter(torch.nn.Module):
    def __init__(self, adv_eps=0.05, iters=5) -> None:
        super().__init__()
        
        self.adv_eps = float(adv_eps)
        self.iters = int(iters)
        self.alpha = adv_eps / 4
    
    def forward(self, model, feats, loss_function, labels):
        delta = torch.rand(feats.shape, device=feats.device) * self.adv_eps * 2 - self.adv_eps
        delta.requires_grad_(True)

        for _ in range(self.iters):
            p_feats = feats + delta

            _, logits, aux_loss = model(p_feats)
            out = logits.log_softmax(dim=1)
            loss = loss_function(out, labels) + aux_loss

            # delta update
            delta_grad = torch.autograd.grad(loss, delta)[0]
            delta.data = torch.clamp(delta.data + self.alpha * delta_grad.sign(), min=-self.adv_eps, max=self.adv_eps).detach()
        adv_feats = feats + delta.detach()
        return adv_feats
        

def get_configs(config_path, data_type, teacher_type, student_type, setting):
    with open(config_path, "r") as conf:
        configs = yaml.load(conf, Loader=yaml.FullLoader)
    try:
        configs = configs[data_type.value][setting.value][teacher_type.value][student_type.value]
    except:
        raise ValueError(f"Invalid config for datatype: ({data_type})\nteacher({teacher_type})\nstudent({student_type})")
    teacher_model_config = configs["teacher"]
    student_model_config = configs["student"]
    teacher_optimizer_config = configs["optimizer"]["teacher"]
    student_optimizer_config = configs["optimizer"]["student"]
    positional_encoding_config = configs["positional_encoding"]
    loss_config = configs["loss"]
    reliable_sampling_config = configs["reliable_sampling"]
    return teacher_model_config, teacher_optimizer_config, \
        student_model_config, student_optimizer_config, \
            positional_encoding_config, loss_config, reliable_sampling_config


def get_positional_encoding(g, positional_encodig_config, data_dir, setting):
    force = False
    if setting is Setting.INDUCTIVE:
        data_dir = None
        force = True
    model_emb = DeepWalk(g, data_dir=data_dir, walk_length=positional_encodig_config["walk_length"], \
                    num_walks=positional_encodig_config["num_walks"])
    # get embedding vectors
    embeddings = model_emb.get_embeddings(window_size=positional_encodig_config["window_size"], \
                iter=positional_encodig_config["iter"], embed_size=positional_encodig_config["emb_size"], \
                force=force, workers=1)    
    return embeddings


def get_label_propagation(graph, labels, train_idx, alpha=0.1, cutoff=10):
    conv = dgl.nn.pytorch.conv.GraphConv(0, 0, norm='right', weight=False, bias=False, allow_zero_in_degree=True)
    num_nodes = graph.num_nodes()
    label_dim = labels.int().max().item() + 1
    
    l_prop = torch.zeros((num_nodes, label_dim), device=graph.device).float()
    l_prop[train_idx] = torch.nn.functional.one_hot(labels[train_idx], num_classes=label_dim).float()

    l_prop = conv(graph, alpha * l_prop)
    for i in range(cutoff-1):
        l_prop += conv(graph, alpha * l_prop)
    l_prop /= torch.sum(l_prop,dim=1, keepdim=True) + 1e-6
    
    return l_prop


def accuracy(out, labels):
    pred = out.argmax(dim=-1)
    return (pred == labels).float().mean().item()


def train_teacher(g, feats, labels, setting, indices, model, loss_func, optimizer, max_epochs, device, verbose):
    if setting is Setting.TRANSDUCTIVE:
        idx_train, idx_val, idx_test = indices
        g_train = g.to(device)
        g_test = g_train
        feats_train = feats.to(device)
        feats_test = feats_train
        labels_train = labels.to(device)
        labels_test = labels_train
    else:
        idx_train, idx_val, idx_test, idx_obs = indices
        g_test = g.to(device)
        g_train = g.subgraph(idx_obs).to(device)
        feats_train = feats[idx_obs].to(device)
        feats_test = feats.to(device)
        labels_train = labels[idx_obs].to(device)
        labels_test = labels.to(device)
    
    model = model.to(device)
    
    es = 0
    val_best = 0
    test_best = 0
    for epoch in range(max_epochs):
        model.train()

        _, logits = model(g_train, feats_train)
        out = logits.log_softmax(dim=1)
        loss = loss_func(out[idx_train], labels_train[idx_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        
        model.eval()
        with torch.no_grad():
            _, logits = model(g_train, feats_train)
            out = logits.log_softmax(dim=1)

            train_acc = accuracy(out[idx_train], labels_train[idx_train])
            val_acc = accuracy(out[idx_val], labels_train[idx_val])

            _, logits = model(g_test, feats_test)
            out = logits.log_softmax(dim=1)
            test_acc = accuracy(out[idx_test], labels_test[idx_test])
        
        if verbose:
            print("Teacher[{}] Train Loss: {:.5f} | Train Acc: {:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f} | Val Best: {:.4f}, Test Best: {:.4f}".format(
                                        epoch, train_loss, train_acc, val_acc, test_acc, val_best, test_best))
        
        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            
        if es == 50:
            if verbose:
                print("Early stopping!")
            break
    
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        emb, logits = model(g_train, feats_train)
        out = logits.log_softmax(dim=1)
        
    return out.detach(), emb.detach(), test_best, state
    

def train_student(g, feats, out_teacher, labels, batch_size, setting, indices, model, \
            loss_function, optimizer, device, max_epochs=100, pretrain_epoch=0, \
                KD_sampler=None, adv_augmenter=None, emb_teacher=None, patience=50, verbose=False):
    if setting is Setting.TRANSDUCTIVE:
        idx_train, idx_val, idx_test = indices
        feats_train = feats.to(device)
        feats_test = feats_train.to(device)
        labels_train = labels.to(device)
        labels_test = labels_train.to(device)     
    else:
        idx_train, idx_val, idx_test, idx_obs = indices
        g = g.to(device)
        feats_train = feats[idx_obs].to(device)
        feats_test = feats.to(device)
        labels_train = labels[idx_obs].to(device)
        labels_test = labels.to(device)
    
    out_teacher = out_teacher.to(device)
    edge_idx_no_loops = dgl.remove_self_loop(g).adjacency_matrix().T.indices().to(device)
    edge_idx_loops = torch.arange(g.num_nodes(), device=device).tile(2,1)
    if emb_teacher is not None:
        emb_teacher = emb_teacher.to(device)
    model = model.to(device)
    
    es = -pretrain_epoch
    val_best = 0
    test_best = 0
    time_eval = []
    for epoch in range(max_epochs+pretrain_epoch):
        if epoch == pretrain_epoch:
            model.finish_pretraining(feats_train, batch=batch_size)
        
        model.train()
        
        edge_idx = edge_idx_loops
        # edge_idx = torch.concat([edge_idx_no_loops, edge_idx_loops], dim=1)
        if KD_sampler is not None:
            edge_idx = edge_idx_no_loops
            KD_prob = torch.from_numpy(KD_sampler.predict_proba()).to(device)
            sampling_mask = torch.bernoulli(KD_prob[edge_idx[1]]).bool()
            edge_idx = torch.masked_select(edge_idx, sampling_mask).view(2, -1).detach()
            edge_idx = torch.concat([edge_idx, edge_idx_loops], dim=1).to(device)

        train_loss = 0
        optimizer.zero_grad()
        # batched classification update (+ similarity)
        for idx in range(0, idx_train.shape[0], batch_size):
            # classification
            batch_idx = idx_train[idx:idx+batch_size]
            batch_feats = feats_train[batch_idx]
            batch_labels = labels_train[batch_idx]
            
            emb, logits, aux_loss = model(batch_feats)
            out = logits.log_softmax(dim=1)
            loss = loss_function.classif_loss(out, batch_labels) + aux_loss
            if emb_teacher is not None:
                # similarity
                similarity_distill_teacher = emb_teacher[batch_idx]
                similarity_distill_matrix = model.encode_mlp4kd(emb)
                similarity_distill_matrix = torch.matmul(similarity_distill_matrix, similarity_distill_matrix.T)
                teacher_sim_matrix = torch.matmul(similarity_distill_teacher, similarity_distill_teacher.T)
                loss += loss_function.feature_sim_loss(similarity_distill_matrix, teacher_sim_matrix)
            loss.backward()
            train_loss += loss.item()
        
        # batched distillation update
        for idx in range(0, edge_idx.shape[1], batch_size):
            batch_idx = edge_idx[:,idx:idx+batch_size]
            if batch_idx.shape[1]<2:
                continue
            batch_feats = feats_train[batch_idx[0]]
            batch_labels = out_teacher[batch_idx[1]]
            emb, logits, aux_loss = model(batch_feats)
            out = logits.log_softmax(dim=1)
            loss = loss_function.distill_loss(out, batch_labels) + aux_loss
            loss.backward()
            train_loss += loss.item()
        
        # batched augmentation update
        if adv_augmenter is not None:
            # classification
            for idx in range(0, idx_train.shape[0], batch_size):
                batch_idx = idx_train[idx:idx+batch_size]
                batch_feats = feats_train[batch_idx]
                batch_labels = labels_train[batch_idx]
                loss = loss_function.augment_classif_loss(model, batch_feats, batch_labels)
                loss.backward()
                train_loss += loss.item()
                
            # distillation
            for idx in range(0, edge_idx.shape[1], batch_size):
                batch_idx = edge_idx[:,idx:idx+batch_size]
                if batch_idx.shape[1]<2:
                    continue
                batch_feats = feats_train[batch_idx[0]]
                batch_labels = out_teacher[batch_idx[1]]
                loss = loss_function.augment_distill_loss(model, batch_feats, batch_labels)
                loss.backward()
                train_loss += loss.item()
                
        # optimizer.zero_grad()
        # loss.backward()
        optimizer.step()
        # train_loss = loss.item()
        
        model.eval()
        
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = [model(feats_test[i:i+batch_size], emb_loss=False)[1]   for i in range(0, feats_test.shape[0], batch_size)]
            logits = torch.concat(logits, dim=0)
            out = logits.log_softmax(dim=1)
        elapsed_time = time.perf_counter() - start_time
        time_eval.append(elapsed_time)
        test_acc = accuracy(out[idx_test], labels_test[idx_test])
        if setting is Setting.INDUCTIVE:
            out = out[idx_obs]
        train_acc = accuracy(out[idx_train], labels_train[idx_train])
        val_acc = accuracy(out[idx_val], labels_train[idx_val])
        
        if KD_sampler is not None:
            KD_sampler.update(out_teacher.detach().cpu().numpy(), out.detach().cpu().numpy())
        
        del out
        del logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        if verbose:
            print("Student[{}] Train Loss: {:.5f} | Train Acc: {:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f} | Val Best: {:.4f}, Test Best: {:.4f}".format(
                                        epoch, train_loss, train_acc, val_acc, test_acc, val_best, test_best))
        
        if test_acc > test_best:
            test_best = test_acc

        if val_acc >= val_best:
            val_best = val_acc
            state = copy.deepcopy(model.state_dict())
            es = 0
        else:
            es += 1
            
        if es == patience:
            if verbose:
                print("Early stopping!")
            break
        
    model.load_state_dict(state)
    model.eval()
    time_eval = np.mean(elapsed_time)
    return test_best, state, time_eval


def graph_split(idx_train, idx_val, idx_test, labels, data_type, split_rate):
    if data_type is DataType.CORA or data_type is DataType.CITESEER or data_type is DataType.PUBMED:
        idx_test_ind = idx_test
        idx_test_tran = torch.tensor(list(set(torch.randperm(labels.shape[0]).tolist()) - set(idx_train.tolist()) - set(idx_val.tolist()) - set(idx_test_ind.tolist())))
    else:
        n = len(idx_test)
        cut = int(n * split_rate)
        idx_idx_shuffle = torch.randperm(n)

        idx_test_ind_idx, idx_test_tran_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
        idx_test_ind, idx_test_tran = idx_test[idx_test_ind_idx], idx_test[idx_test_tran_idx]

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    N1, N2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:N1]
    obs_idx_val = obs_idx_all[N1 : N1 + N2]

    return obs_idx_train, obs_idx_val, idx_test_ind, idx_obs


def update_test_positional_encoding(g, obs_idx, position_feature):
    updated_features = torch.zeros((g.num_nodes(),position_feature.shape[1]),dtype=position_feature.dtype,device=position_feature.device)
    updated_features[obs_idx] = position_feature
    
    with g.local_scope():
        g.ndata["emb"] = updated_features
        idx = g.filter_nodes(lambda x: torch.sum(x.data["emb"], dim=1)==0)
        g.pull(idx, dgl.function.copy_u('emb', 'm'), dgl.function.mean('m', 'emb'))

        return g.ndata["emb"]
    
    # def copier(edges):
    #     return {'m': edges.src['emb'], "o": edges.src['obs']}
    
    # def reducer(nodes):
    #     m = nodes.mailbox["m"]
    #     o = nodes.mailbox["o"]
    #     den = (o>0).float().sum(1, keepdim=True)
    #     m = torch.sum(m, dim=1)
    #     out = m / (den + 1e-10)
    #     return {"emb": out}
    
    # with g.local_scope():
    #     g.ndata["emb"] = updated_features
    #     g.ndata["obs"] = torch.sum(updated_features, dim=1)
    #     idx = g.filter_nodes(lambda x: x.data["obs"]==0)
    #     g.pull(idx, copier, reducer)

    #     return g.ndata["emb"]
    

def main(data_type, data_dir, setting, batch_size, teacher_type, student_type, config_path,\
            use_reliable_sampling, use_positional_encoding, use_label_propagation, \
                use_similarity_distill, use_adv_augment, seed, split_rate, prefix, num_runs, \
                    patience, only_teacher, device, verbose):
    fix_seed = seed >= 0
    test_best_teacher_list = []
    test_best_student_list = []
    student_time_eval = []
    for s in range(num_runs):
        if fix_seed:
            seed = s
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            dgl.random.seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        else:
            seed = None
        
        g, labels, idx_train, idx_val, idx_test = load_data(data_type, data_dir, seed, verbose=verbose)
        if setting is Setting.TRANSDUCTIVE:
            indices = (idx_train, idx_val, idx_test)
        else:
            indices = graph_split(idx_train, idx_val, idx_test, labels, data_type, split_rate)
        feats_teacher = g.ndata["feat"]
        feat_dim_teacher = feats_teacher.shape[1]
        label_dim = labels.int().max().item() + 1
        
        teacher_model_config, teacher_optimizer_config, \
            student_model_config, student_optimizer_config, \
                positional_encoding_config, loss_config, reliable_sampling_config = get_configs(config_path, data_type, teacher_type, student_type, setting)
        
        # Train teacher
        loss_teacher = torch.nn.NLLLoss()
        model_teacher = load_teacher_model(teacher_type, feat_dim_teacher, label_dim, **teacher_model_config)
        optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), \
            lr=float(teacher_optimizer_config["learning_rate"]),\
                weight_decay=float(teacher_optimizer_config["weight_decay"]))
        out_teacher, emb_teacher, test_best_teacher, _ = train_teacher(g, feats_teacher, labels, setting, indices, \
                            model_teacher, loss_teacher, optimizer_teacher, teacher_optimizer_config["max_epoch"], device, verbose)
        
        if verbose:
            print("Clear gpu memory from graph")
        # clear gpu memory from graph
        g = g.cpu()
        labels = labels.cpu()
        feats_teacher = feats_teacher.cpu()
        out_teacher = out_teacher.cpu()
        emb_teacher = emb_teacher.cpu()
        model_teacher = model_teacher.cpu()
        del loss_teacher
        del optimizer_teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        test_best_teacher_list.append(test_best_teacher)
        if only_teacher:
            test_best_student_list.append(0.0)
            continue
        
        # student training graph
        if setting is Setting.INDUCTIVE:
            student_g = g.subgraph(indices[3])
        else:
            student_g = g
        
        # Positional encodings
        feats_student = feats_teacher
        feat_dim_student = feat_dim_teacher
        if use_positional_encoding:
            if verbose:
                print("Generating positional encodings")
            
            data_dir_ = os.path.join(data_dir, f"graph_emb_{data_type.value}")
            position_feature = get_positional_encoding(student_g, positional_encoding_config, data_dir_, setting)
            if setting is Setting.INDUCTIVE:
                # update target nodes with the mean value of existing neighbours
                position_feature = update_test_positional_encoding(g, indices[3], position_feature)
            feats_student = torch.cat([feats_student, position_feature], dim=1)
            feat_dim_student = feats_student.shape[1]
        if use_label_propagation:
            label_propagation = get_label_propagation(g, labels, indices[0], alpha=0.1)
            feats_student = torch.cat([feats_student, label_propagation], dim=1)
            feat_dim_student = feats_student.shape[1]
        
        if batch_size<0:
            batch_size = feats_student.shape[0]
        
        # Initialize sampler
        KD_sampler = None
        if use_reliable_sampling:
            if verbose:
                print("Initializing sampler")
            if setting is Setting.TRANSDUCTIVE:
                sampler_feats = feats_teacher
            else:
                sampler_feats = feats_teacher[indices[3]]
            KD_sampler = Com_KD_Prob(model_teacher, student_g, sampler_feats, **reliable_sampling_config)

        # Initialize augmenter
        adv_augmenter = None
        if use_adv_augment:
            if verbose:
                print("Initializing augmenter")
            adv_augmenter = AdversarialFeatureAugmenter(loss_config["adv_eps"], loss_config["adv_iters"])
            
        # Similarity distillation
        if not use_similarity_distill:
            emb_teacher = None
        
        if verbose:
            print("Training student")
        # Train student
        loss_student = CombinedClassificationDistillationLoss(loss_config["lambda"], loss_config["tau"], adv_augmenter, loss_config["adv_weigth"], loss_config["feat_distill_weight"],)
        model_student = load_student_model(student_type, feat_dim_student, label_dim, **student_model_config)
        optimizer_student = torch.optim.Adam(model_student.parameters(), \
            lr=float(student_optimizer_config["learning_rate"]),\
                weight_decay=float(student_optimizer_config["weight_decay"]))
        
            
        test_best_student, _, time_eval  = train_student(student_g, feats_student, out_teacher, labels, batch_size, setting, \
                    indices, model_student, loss_student, optimizer_student, device, student_optimizer_config["max_epoch"], \
                        student_optimizer_config["pretrain_epoch"], KD_sampler, adv_augmenter, emb_teacher, patience, verbose)
        
        test_best_student_list.append(test_best_student)
        student_time_eval.append(time_eval)
    
    mean_teacher = np.mean(test_best_teacher_list)*100
    mean_student = np.mean(test_best_student_list)*100
    std_teacher = np.std(test_best_teacher_list)*100
    std_student = np.std(test_best_student_list)*100
    mean_eval_time = np.mean(student_time_eval)
    
    if len(prefix)>0:
        prefix = f"{prefix} | "
    print(f"{prefix}{data_type.value} | Teacher: {mean_teacher:.2f} ± {std_teacher:.2f} | Student: {mean_student:.2f} ± {std_student:.2f} | Eval time: {mean_eval_time} s")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("distill2moe")
    parser.add_argument("-d", "--datatype", type=DataType, default=DataType.CORA, choices=list(DataType))
    parser.add_argument("-m", "--mode", type=Setting, default=Setting.TRANSDUCTIVE, choices=list(Setting))
    parser.add_argument("-t", "--teacher_type", type=TeacherModelType, default=TeacherModelType.GCN, choices=list(TeacherModelType))
    parser.add_argument("-s", "--student_type", type=StudentModelType, default=StudentModelType.MLP, choices=list(StudentModelType))
    parser.add_argument("--config", type=str, default="configs/parameters.yaml")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--reliable_sampling", action="store_true")
    parser.add_argument("--positional_encoding", action="store_true")
    parser.add_argument("--label_propagation", action="store_true")
    parser.add_argument("--similarity_distill", action="store_true")
    parser.add_argument("--adv_augment", action="store_true")
    parser.add_argument("--only_teacher", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_rate", type=float, default=0.2)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--gpu_id", type=int, default=-1)
    args = parser.parse_args()
    
    datatype = args.datatype
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    setting = args.mode
    teacher_type = args.teacher_type
    student_type = args.student_type
    config_path = args.config
    assert os.path.isfile(config_path)
    batch_size = args.batch_size
    assert batch_size > 0 or batch_size == -1
    use_reliable_sampling = args.reliable_sampling
    use_positional_encoding = args.positional_encoding
    use_label_propagation = args.label_propagation
    use_similarity_distill = args.similarity_distill
    use_adv_augment = args.adv_augment
    verbose = args.verbose
    only_teacher = args.only_teacher
    seed = args.seed
    split_rate = args.split_rate
    assert 0 < split_rate < 1
    prefix = args.prefix
    num_runs = args.num_runs
    assert num_runs > 0
    patience = args.patience
    assert patience > 1
    gpu_id = args.gpu_id
    if gpu_id<0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    main(datatype, data_dir, setting, batch_size, teacher_type,  student_type, config_path, use_reliable_sampling, \
            use_positional_encoding, use_label_propagation, use_similarity_distill, use_adv_augment, seed, split_rate, \
                prefix, num_runs, patience, only_teacher, device, verbose)
