import enum
import os
import torch
import numpy as np
from dgl.nn.pytorch import GraphConv, SAGEConv
from scipy.stats import entropy
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans



class TeacherModelType(enum.Enum):
    GCN = "gcn"
    SAGE = "sage"
    
    DRGAT = "drgat"
    RevGAT = "revgat"

    def __str__(self):
        return self.value
    

class StudentModelType(enum.Enum):
    MLP = "mlp"
    MOE = "moe"
    RBM = "rbm"

    def __str__(self):
        return self.value



def load_teacher_model(modeltype: TeacherModelType, input_dim, output_dim, **model_params):
    if modeltype is TeacherModelType.GCN:
        return GCN(input_dim, output_dim, **model_params)
    if modeltype is TeacherModelType.SAGE:
        return GraphSAGE(input_dim, output_dim, **model_params)
    if modeltype in [TeacherModelType.DRGAT, TeacherModelType.RevGAT]:
        return PretrainedGNNTeacher(**model_params)
    raise ValueError(f"Teacher model type {modeltype} is not supported")


def load_student_model(modeltype: StudentModelType, input_dim, output_dim, **model_params):
    if modeltype is StudentModelType.MLP:
        return MLPWrapper(MLP(input_dim, output_dim, **model_params))
    if modeltype is StudentModelType.MOE:
        return MoE(input_dim, output_dim, **model_params)
    if modeltype is StudentModelType.RBM:
        return RbM(input_dim, output_dim, **model_params)
    raise ValueError(f"Student model type {modeltype} is not supported")



class PretrainedGNNTeacher(torch.nn.Module):
    def __init__(self, model_dir_path=None, **kwargs):
        super().__init__()
        
        assert model_dir_path is not None
        
        out_t = torch.from_numpy(np.load(os.path.join(model_dir_path,"out.npz"))["arr_0"])
        self.out_t = torch.softmax(out_t, dim=-1)

    def forward(self, g, feats):
        return [None], self._out_t


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2,
                 hidden_dim=64, dropout=0.0, norm_type=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.dropout = torch.nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        self.use_layer_norm = True
        if norm_type == "batch":
            norm_class = torch.nn.BatchNorm1d
        elif norm_type == "layer":
            norm_class = torch.nn.LayerNorm
        else:
            self.use_layer_norm = False
        
        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=torch.nn.ReLU()))
            return
        self.layers.append(GraphConv(input_dim, hidden_dim, activation=torch.nn.ReLU()))
        if self.use_layer_norm:
            self.norms.append(norm_class(hidden_dim))
        for i in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=torch.nn.ReLU()))
            if self.use_layer_norm:
                self.norms.append(norm_class(hidden_dim))
        self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        h_emb = h
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h_emb = h
                if self.use_layer_norm:
                    h = self.norms[l](h)
                h = self.dropout(h)
        return h_emb, h
        

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2,
                 hidden_dim=64, dropout=0.0, norm_type=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        self.use_layer_norm = True
        if norm_type == "batch":
            norm_class = torch.nn.BatchNorm1d
        elif norm_type == "layer":
            norm_class = torch.nn.LayerNorm
        else:
            self.use_layer_norm = False
        
        
        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, aggregator_type='gcn'))
            return
        self.layers.append(SAGEConv(input_dim, hidden_dim, aggregator_type='gcn'))
        if self.use_layer_norm:
            self.norms.append(norm_class(hidden_dim))
        for i in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='gcn'))
            if self.use_layer_norm:
                self.norms.append(norm_class(hidden_dim))
        self.layers.append(SAGEConv(hidden_dim, output_dim, aggregator_type='gcn'))

    def forward(self, g, feats):
        h = feats
        h_emb = h
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h_emb = h
                if self.use_layer_norm:
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h_emb, h



class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2,
                 hidden_dim=64, dropout=0.0, norm_type=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        self.use_layer_norm = True
        if norm_type == "batch":
            norm_class = lambda d: BatchNorm1dBypassWrapper(torch.nn.BatchNorm1d(d))
        elif norm_type == "layer":
            norm_class = torch.nn.LayerNorm
        else:
            self.use_layer_norm = False
        
        
        if num_layers == 1:
            self.layers.append(torch.nn.Linear(input_dim, output_dim))
            return
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        if self.use_layer_norm:
            self.norms.append(norm_class(hidden_dim))
        for i in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            if self.use_layer_norm:
                self.norms.append(norm_class(hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.mlp_feature_encoder = torch.nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, feats):
        h = feats
        h_emb = h
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l == 0:
               h_emb = h
            if l != self.num_layers - 1:
                # h_emb = h
                if self.use_layer_norm:
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h_emb, h
    
    def encode_mlp4kd(self, mlp_feat):
        return self.dropout(self.activation(self.mlp_feature_encoder(mlp_feat)))



class MLPWrapper(torch.nn.Module):
    def __init__(self, mlp):
        super(MLPWrapper, self).__init__()
        
        self.mlp = mlp
    
    def encode_mlp4kd(self, mlp_feat):
        return self.mlp.encode_mlp4kd(mlp_feat)

    def forward(self, x, *args, **kwargs):
        mlp_emb, mlp_out = self.mlp(x)
        return mlp_emb, mlp_out, 0
    
    def finish_pretraining(self, *args, **kwargs):
        pass


class Com_KD_Prob(object):
    def __init__(self, model, g, feats, min_update_epoch=50, update_rate=1, \
                    init_power=1.0, momentum=0.9, bins_num=40):        
        self.min_update_epoch = min_update_epoch
        self.update_rate = update_rate
        self._update_counter = 0
        self.init_power = init_power
        self.momentum = momentum
        self.bins_num = bins_num

        model.eval()
        with torch.no_grad():
            _, logits_teacher = model.forward(g, feats)
            out_teacher = logits_teacher.softmax(dim=-1).detach().cpu().numpy()
            feats_noise = torch.clone(feats) + torch.randn(feats.shape[0], feats.shape[1]).to(feats.device)
            _, logit_noise = model.forward(g, feats_noise)
            out_noise = logit_noise.softmax(dim=-1).detach().cpu().numpy()

        weight_s = np.abs(entropy(out_noise, axis=1) - entropy(out_teacher, axis=1))
        self._delta_entropy = weight_s / np.max(weight_s)
        self._power = init_power

    def curve_fit(self, xdata, ydata):
        def func(x, a):
            return 1 - x ** a
        popt, pcov = curve_fit(func, xdata, ydata, p0 = [self._power])
        return popt[0]

    def update(self, logits_t, logits_s):
        self._update_counter += 1
        if self._update_counter < self.min_update_epoch:
            return
        if self._update_counter % self.update_rate != 0:
            return
        
        idxs = np.argmax(logits_t,axis=1) == np.argmax(logits_s,axis=1)
        weight_true = self._delta_entropy[idxs]
        weight_false = self._delta_entropy[~idxs]
        
        hist_t, bins_t = np.histogram(weight_true, bins=self.bins_num, range=(0, 1))
        hist_s, _ = np.histogram(weight_false, bins=self.bins_num, range=(0, 1))

        prob = hist_t / (hist_t + hist_s + 1e-6)
        prob = (prob - np.min(prob)) / (np.max(prob) - np.min(prob))
        update_power = self.curve_fit(bins_t[:-1] + 0.05, prob)
        self._power = self.momentum * self._power + (1-self.momentum) * update_power

    def predict_proba(self):
        return 1 - self._delta_entropy ** self._power



class BatchNorm1dBypassWrapper(torch.nn.Module):
    def __init__(self, batch_norm, bypass_batch=True) -> None:
        super(BatchNorm1dBypassWrapper, self).__init__()
        
        self.batch_norm = batch_norm
        self.bypass_batch = bypass_batch
        
    def forward(self, x):
        if x.shape[0] < 2 and self.bypass_batch:
            self.batch_norm.eval()
        out = self.batch_norm(x)
        if self.training:
            self.batch_norm.train()
        return out
    

class Dispatcher(object):
    def __init__(self, expert_mask):
        non_zero = torch.nonzero(expert_mask)
        _, index_sorted_experts = non_zero.sort(0)
        self.batch_size, self.num_experts = expert_mask.shape
        self.batch_index = non_zero[index_sorted_experts[:,1], 0]
        self.part_sizes = (expert_mask > 0).long().sum(0).tolist()
    
    def dispatch(self, input):
        inp_exp = input[self.batch_index]
        inp_exp = torch.split(inp_exp, self.part_sizes, dim=0)
        return inp_exp

    def combine(self, expert_outputs):
        batch_index_list = torch.split(self.batch_index, self.part_sizes, dim=0)
        final_output = torch.zeros((self.batch_size, expert_outputs[0].shape[1]), dtype=expert_outputs[0].dtype, device=expert_outputs[0].device)
        for bi, eo in zip(batch_index_list, expert_outputs):
            final_output = final_output.index_add(0,bi,eo)
        return final_output
    

class MoERouterProportional(torch.nn.Module):
    def __init__(self, num_experts, sparsity=0.1):
        super(MoERouterProportional, self).__init__()

        self.num_experts = num_experts
        self.sparsity = sparsity
        
    def forward(self, x):
        idxs = torch.arange(x.shape[0])
        idxs = torch.tensor_split(idxs, self.num_experts, dim=0)
        expert_mask = torch.zeros((x.shape[0], self.num_experts), device=x.device)
        for i, idx in enumerate(idxs):
            expert_mask[idx, i] = 1
        routes_prob = expert_mask
        importance = expert_mask.sum(0)
        load = routes_prob.sum(0)
        return expert_mask, routes_prob, importance, load
    

class MoERouterRandom(torch.nn.Module):
    def __init__(self, num_experts, k=3):
        super(MoERouterRandom, self).__init__()

        self.num_experts = num_experts
        self.k = k
        
    def forward(self, x):
        routes_prob = torch.rand((x.shape[0], self.num_experts), device=x.device)
        
        top_logits, top_indices = routes_prob.topk(self.k, dim=-1, sorted=True)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        expert_mask = torch.zeros_like(routes_prob).scatter(1, top_k_indices, torch.ones_like(top_k_logits))
        
        routes_prob = torch.softmax(routes_prob, dim=-1)
        importance = expert_mask.sum(0)
        load = routes_prob.sum(0)
        return expert_mask, routes_prob, importance, load


class MoERouterGauss(torch.nn.Module):
    def __init__(self, router, noise_model, num_experts, k):
        super(MoERouterGauss, self).__init__()
        
        self.router = router
        self.noise = noise_model
        self.num_experts = num_experts
        self.k = min(k + 1, self.num_experts)
        self._eps = 1e-2
        self._normal_distribution = torch.distributions.normal.Normal(0.0, 1/self.num_experts)

    def forward(self, x):
        x = torch.flatten(x, 1)
        route_logits = self.router(x)
        raw_logits = route_logits
        if self.training:
            noise_stddev = self.noise(x)
            noise_stddev = torch.nn.functional.softplus(noise_stddev) + self._eps
            noisy_logits = route_logits + torch.randn_like(route_logits) * noise_stddev
            route_logits = noisy_logits
        
        top_logits, top_indices = route_logits.topk(self.k, dim=-1, sorted=True)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        expert_mask = torch.zeros_like(route_logits).scatter(1, top_k_indices, torch.ones_like(top_k_logits))
        
        if self.training:
            gumbels = (-torch.empty_like(route_logits)).exponential_().log() # ~Gumbel(0,1)
            gumbels = route_logits + gumbels  # ~Gumbel(logits, 1)
            mask_soft = torch.softmax(gumbels, dim=-1)
            expert_mask = expert_mask - mask_soft.detach() + mask_soft
            
        route_prob = torch.softmax(route_logits, dim=-1)      
        
        # importance = expert_mask.sum(0)
        importance = torch.softmax(raw_logits, dim=-1).sum(0)
        if self.training:
            # Get the smallest routing weight of a selected expert at each point.
            threshold = noisy_logits.topk(self.k, dim=-1, sorted=True).values[:, self.k-1]
            
            # Find how far each routing weight is from this threshold.
            distance_to_selection = (threshold.unsqueeze(1) - raw_logits)/noise_stddev
            # Compute the probability that, if we resampled the noise, an
            # expert would be selected.
            p = 1.0 - self._normal_distribution.cdf(distance_to_selection)
            # Compute the load over all samples.
            load = p.sum(dim=0)
        else:
            load = route_prob.sum(0)
        return expert_mask, route_prob, importance, load
    
    

class MoERouterSwitch(torch.nn.Module):
    def __init__(self, router, num_experts, k, non_expert_prob):
        super(MoERouterSwitch, self).__init__()
        
        self.router = router
        self.num_experts = num_experts
        self.k = min(k + 1, self.num_experts)
        self.non_expert_prob = non_expert_prob

    def forward(self, x):
        x = torch.flatten(x, 1)
        route_logits = self.router(x)
        raw_logits = route_logits
        if self.training:
            top_logits, top_indices = route_logits.topk(self.k, dim=-1)
            top_logit_mask = torch.zeros_like(route_logits).scatter(1, top_indices, top_logits)
            use_other_expert_mask = (torch.rand_like(top_logits, device=x.device)<self.non_expert_prob).float().sum(1)>0 # BxK
            route_logits[use_other_expert_mask] -= top_logit_mask[use_other_expert_mask]
            
        top_logits, top_indices = route_logits.topk(self.k, dim=-1)
        expert_mask = torch.zeros_like(route_logits).scatter(1, top_indices, torch.ones_like(top_logits))
        route_prob = torch.softmax(route_logits, dim=-1)
        
        # importance = expert_mask.sum(0)
        importance = torch.softmax(raw_logits, dim=-1).sum(0)
        load = route_prob.sum(0)
        return expert_mask, route_prob, importance, load



class MoELayerRouterModel(torch.nn.Module):
    def __init__(self, num_experts, k, input_dim, hidden_dim=256):
        super(MoELayerRouterModel, self).__init__()
        
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = min(k + 1, self.num_experts)
        
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 2)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(input_dim, hidden_dim)
        self.sim_matrix = torch.nn.Parameter(torch.empty((hidden_dim, num_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(100)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)
        
    def forward(self, x):
        logits = torch.matmul(torch.nn.functional.normalize(self.cosine_projector(x), dim=1),
                              torch.nn.functional.normalize(self.sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        
        if self.training:
            logits = logits + torch.randn_like(logits) / self.num_experts
            
        top_logits, top_indices = logits.topk(self.k, dim=-1)
        expert_mask = torch.zeros_like(logits).scatter(1, top_indices, torch.ones_like(top_logits))
        route_prob = torch.softmax(logits, dim=-1)
        
        importance = route_prob.sum(0)
        load = expert_mask.sum(0)
        
        return expert_mask, route_prob, importance, load



class MoELayer(torch.nn.Module):
    def __init__(self, expert_builder, router, num_experts=8, k=3, load_balance_coef=0.01, use_gshard_balance=False):
        super(MoELayer, self).__init__()
        
        assert k <= num_experts
        
        self.router = router
        self.pretrain_experts_router = MoERouterProportional(num_experts)
        self.num_experts = num_experts
        self.k = k
        self.load_balance_coef = load_balance_coef
        self.use_gshard_balance = use_gshard_balance
        
        # instantiate experts
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 2)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(100)).item()
        self.experts = torch.nn.ModuleList([
            expert_builder()
            for _ in range(self.num_experts) ])
        self.device = torch.device("cpu")
        self.pretraining = True
    
    def forward(self, x, emb_loss=True):
        if self.pretraining:
            router = self.pretrain_experts_router
        else:
            router = self.router
        expert_mask, routes_prob, importance, load = router(x) 
        # expert_mask BxE
        # routes_prob BxE
        
        dispatcher = Dispatcher(expert_mask)
        x_exp = dispatcher.dispatch(x)
        route_prob_exp = dispatcher.dispatch(routes_prob)
        route_prob_exp = [route_prob_exp[i][:,[i]] for i in range(len(route_prob_exp))]  # [B'x1]
        expert_outputs = [rp*exp(inp) for exp, inp, rp in zip(self.experts, x_exp, route_prob_exp)]
        final_output = dispatcher.combine(expert_outputs)
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        final_output = final_output * logit_scale
        
        if self.pretraining or not emb_loss:
            aux_loss = 0
        else:
            aux_loss = self.balance_loss(final_output.shape[0], importance, load)
        
        return final_output, aux_loss

    def balance_loss(self, batch_size, importance, load):
        if self.use_gshard_balance:
            out = torch.sum(importance*load) / batch_size
        else:
            out = cv_squared(importance) + cv_squared(load)
        return self.load_balance_coef * out
    
    def finish_pretraining(self):
        self.pretraining = False
        


class MoE(torch.nn.Module):
    """Mixture of Experts"""
    def __init__(self, input_dim, output_dim, num_layers=2,
                hidden_dim=64, dropout=0.0, norm_type=None,
                num_experts=8, k=3, load_balance_coef=0.01):
        super(MoE, self).__init__()
        
        assert k < num_experts
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = k
        self.load_balance_coef = load_balance_coef
        
        self.pretraining = True
        
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        self.mlp_feature_encoder = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.use_layer_norm = True
        if norm_type == "batch":
            norm_class = lambda d: BatchNorm1dBypassWrapper(torch.nn.BatchNorm1d(d))
        elif norm_type == "layer":
            norm_class = torch.nn.LayerNorm
        else:
            self.use_layer_norm = False
        
        # instantiate moe layers
        if num_layers == 1:
            self.layers.append(self._build_layer(input_dim, output_dim))
            return
        self.layers.append(self._build_layer(input_dim, hidden_dim))
        if self.use_layer_norm:
            self.norms.append(norm_class(hidden_dim))
        for i in range(num_layers - 2):
            self.layers.append(self._build_layer(hidden_dim, hidden_dim))
            if self.use_layer_norm:
                self.norms.append(norm_class(hidden_dim))
        self.layers.append(self._build_layer(hidden_dim, output_dim))
        
    def _build_layer(self, input_dim, output_dim):
        model_builder = lambda: torch.nn.Sequential(self.dropout,
                                                    torch.nn.Linear(input_dim, output_dim))
        router = MoELayerRouterModel(self.num_experts, self.k, input_dim, self.hidden_dim)
        # if self.use_switch:
        #     router_model = torch.nn.Linear(input_dim, self.num_experts)
        #     router = MoERouterSwitch(router_model, self.num_experts, self.k, self.non_expert_prob)
        # else:
        #     router_model = torch.nn.Linear(input_dim, self.num_experts)
        #     noise_model = torch.nn.Linear(input_dim, self.num_experts)
        #     router = MoERouterGauss(router_model, noise_model, self.num_experts, self.k)
        return MoELayer(model_builder, router, self.num_experts, self.k, self.load_balance_coef)

    def forward(self, x, emb_loss=True):
        h = x
        h_emb = h
        aux_loss = 0
        for l, layer in enumerate(self.layers):
            h, al = layer(h, emb_loss)
            aux_loss = aux_loss + al
            if l == 0:
               h_emb = h
            if l != self.num_layers - 1:
                # h_emb = h
                if self.use_layer_norm:
                    h = self.norms[l](h)
                h = self.activation(h)
                # h = self.dropout(h)
        
        return h_emb, h, aux_loss
    
    def encode_mlp4kd(self, mlp_feat):
        return self.dropout(self.activation(self.mlp_feature_encoder(mlp_feat)))
    
    def finish_pretraining(self, *args, **kwargs):
        for l in self.layers:
            l.finish_pretraining()



class RbMLayer(torch.nn.Module):
    def __init__(self, expert_builder, init_memory, num_experts=8, k=3, momentum=0.9,\
                    vq_coeff=0.05, sim_coeff=0.01, load_balance_coef=0.01, input_attention=False):
        super(RbMLayer, self).__init__()
        
        assert k <= num_experts
        
        self.num_experts = num_experts
        self.k = k
        self.momentum = momentum
        self.vq_coeff = vq_coeff
        self.sim_coeff = sim_coeff
        self.load_balance_coef = load_balance_coef
        self.input_attention = input_attention
        self.routing_memory = torch.nn.Parameter(init_memory, requires_grad=True)
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 2)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(100)).item()
        self.const_attention = torch.nn.Parameter(torch.zeros([num_experts]), requires_grad=False)
        if input_attention:
            self.const_attention = torch.nn.Parameter(torch.log(torch.full([num_experts, init_memory.shape[1]], 1.1)), requires_grad=True)
        self.experts = torch.nn.ModuleList([
            expert_builder()
            for _ in range(self.num_experts)
        ])
        self.pretraining = True
    
    def forward(self, x, emb_loss=True):
        if self.pretraining:
            output = self.experts[0](x)
            return output, 0
        
        routing_memory = torch.nn.functional.normalize(self.routing_memory, dim=1).detach()
        norm_x = torch.nn.functional.normalize(x, dim=1)
        dists = torch.matmul(norm_x, routing_memory.T)
        
        top_values, top_indices = dists.topk(self.k, dim=-1, largest=True, sorted=True)
        top_k_values = top_values[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        expert_mask = torch.zeros_like(dists).scatter(1, top_k_indices, 1.0)
        top_k_values = torch.softmax(top_k_values, dim=-1)
        attention_w = torch.zeros_like(dists).scatter(1, top_k_indices, top_k_values)
        
        # dispatching
        dispatcher = Dispatcher(expert_mask)
        x_exp = dispatcher.dispatch(x)
        a_exp = dispatcher.dispatch(attention_w)
        inp_attention = torch.clamp(self.const_attention, max=self.clamp_max).exp()
        exp_outputs = []
        for i, (e, inp, a) in enumerate(zip(self.experts, x_exp, a_exp)):
            o = e(inp_attention[[i]]*inp)
            att = a[:,[i]]
            exp_outputs.append(att.detach()*o)

        # collecting
        final_output = dispatcher.combine(exp_outputs)
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        final_output = final_output * logit_scale
        
        # state update
        if self.training:
            with torch.no_grad():
                avgs = []
                for i, (inp, a) in enumerate(zip(x_exp, a_exp)):
                    inp = torch.nn.functional.normalize(inp, dim=1)
                    att = a[:,[i]]
                    att = torch.softmax(att, dim=0)
                    avgs.append((att*inp).sum(0))
                avgs = torch.stack(avgs) # ExN
                mov_avgs = self.momentum*routing_memory + (1-self.momentum)*avgs
                self.routing_memory.data = mov_avgs
        
        if not emb_loss:
            return final_output, 0
        
        aux_loss = 0
        ### VQ-style loss
        if self.vq_coeff > 0:
            vq_loss = 0
            for i in range(self.k):
                top_indices = top_k_indices[:, i]
                vq_ = -norm_x * routing_memory[top_indices]
                vq_loss += top_k_values[:, i] * vq_.sum(-1)
            aux_loss += self.vq_coeff*vq_loss.mean()
        
        ### similarity loss
        if self.sim_coeff > 0:
            routing_memory = torch.nn.functional.normalize(self.routing_memory, dim=1)
            sim_loss = torch.matmul(routing_memory, routing_memory.T.detach()).mean()
            aux_loss += self.sim_coeff*sim_loss
        
        ### load balance loss
        if self.load_balance_coef > 0:
            load_balance = attention_w.sum(0)
            load_balance = cv_squared(load_balance)
            aux_loss += self.load_balance_coef*load_balance
        
        return final_output, aux_loss
        
    
    def finish_pretraining(self, feats):
        feats = torch.nn.functional.normalize(feats, dim=1)
        kmean = KMeans(self.num_experts).fit(feats.detach().cpu().numpy())
        centers = torch.from_numpy(kmean.cluster_centers_).to(self.routing_memory.device)
        self.routing_memory.data = torch.nn.functional.normalize(centers, dim=1)
        for i in range(1,len(self.experts)):
            self.experts[i].load_state_dict(self.experts[0].state_dict())
        self.pretraining = False



class RbM(torch.nn.Module):
    """Routing by Memory"""
    def __init__(self, input_dim, output_dim, num_layers=2,
                hidden_dim=64, dropout=0.0, norm_type=None,
                num_experts=8, k=3, momentum=0.9, vq_coeff=0.05, 
                load_balance_coef=0.01, sim_coeff=0.0, input_attention=False):
        super(RbM, self).__init__()
        
        assert k < num_experts
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = k
        self.momentum = momentum
        self.vq_coeff = vq_coeff
        self.sim_coeff = sim_coeff
        self.load_balance_coef = load_balance_coef
        self.input_attention = input_attention
        
        self.pretraining = True
        
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        self.mlp_feature_encoder = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.use_layer_norm = True
        if norm_type == "batch":
            norm_class = lambda d: BatchNorm1dBypassWrapper(torch.nn.BatchNorm1d(d))
        elif norm_type == "layer":
            norm_class = torch.nn.LayerNorm
        else:
            self.use_layer_norm = False
        
        # instantiate moe layers
        if num_layers == 1:
            self.layers.append(self._build_layer(self.input_dim, self.output_dim))
            return
        self.layers.append(self._build_layer(self.input_dim, self.hidden_dim))
        if self.use_layer_norm:
                self.norms.append(norm_class(self.hidden_dim))
        for i in range(num_layers - 2):
            self.layers.append(self._build_layer(self.hidden_dim, self.hidden_dim))
            if self.use_layer_norm:
                self.norms.append(norm_class(self.hidden_dim))
        self.layers.append(self._build_layer(self.hidden_dim, self.output_dim))
        
    def _build_layer(self, input_dim, output_dim):
        model_builder = lambda: torch.nn.Sequential(self.dropout,
                                                    torch.nn.Linear(input_dim, output_dim))
        memory = torch.zeros(self.num_experts, input_dim, dtype=torch.float32)
        return RbMLayer(model_builder, memory, self.num_experts, self.k,
                        self.momentum, self.vq_coeff, self.sim_coeff, 
                        self.load_balance_coef, self.input_attention)
        
    def forward(self, x, emb_loss=True):
        h = x
        h_emb = h
        aux_loss = 0
        for l, layer in enumerate(self.layers):
            h, al = layer(h, emb_loss)
            aux_loss = aux_loss + al
            if l == 0:
               h_emb = h
            if l != self.num_layers - 1:
                # h_emb = h
                if self.use_layer_norm:
                    h = self.norms[l](h)
                h = self.activation(h)
                # h = self.dropout(h)
        
        return h_emb, h, aux_loss
    
    def encode_mlp4kd(self, mlp_feat):
        return self.dropout(self.activation(self.mlp_feature_encoder(mlp_feat)))
    
    def finish_pretraining(self, feats, batch=-1):
        self.eval()
        with torch.no_grad():
            h = feats
            for l, layer in enumerate(self.layers):
                layer.finish_pretraining(h)
                if batch > 0:
                    h_new = [layer(h[i:i+batch], False)[0]   for i in range(0, h.shape[0], batch)]
                    h = torch.concat(h_new, dim=0)
                else:
                    h, _ = layer(h, False)
                if l != self.num_layers - 1:
                    if self.use_layer_norm:
                        h = self.norms[l](h)
                    h = self.activation(h)
                    # h = self.dropout(h)
        self.train()
    

def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean()**2 + eps)