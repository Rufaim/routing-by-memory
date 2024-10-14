import os
import numpy as np
import enum
import networkx as nx
import torch
from .walker import RandomWalker
from gensim.models import Word2Vec
import dgl
from abc import ABC, abstractmethod, abstractproperty
from collections import Counter



class PosEncType(enum.Enum):
    DEEPWALK = 'deepwalk'
    RRWP = "rrwp"
    INVDISTATT = "inv_dist_att"
    RWATT = "rw_att"
    LRRWP = "mlp_rrwp"
    DISTEMB = "dist_emb_sum"
    LDISTEMB = "mlp_dist_emb"

    def __str__(self):
        return self.value
    
    def is_learnable(self):
        if self in [PosEncType.LRRWP, PosEncType.DISTEMB, PosEncType.DISTEMBATT,
                    PosEncType.LDISTEMB, PosEncType.LDISTEMBATT]:
            return True
        return False
    

def get_positional_encoder(enc_type:PosEncType, positional_encodig_config, data_dir,
                           is_inductive=False):
    if enc_type is PosEncType.DEEPWALK:
        preprocessor = DeepWalkPreprocessor
        concater = EmbeddingsConcater
    elif enc_type is PosEncType.RRWP:
        preprocessor = RRWPPreprocessor
        concater = EmbeddingsConcater
    elif enc_type is PosEncType.INVDISTATT:
        preprocessor = InverseDistanceAttentionPreprocessor
        concater = EmbeddingsConcater
    elif enc_type is PosEncType.RWATT:
        preprocessor = RandomWalkAttentionPreprocessor
        concater = EmbeddingsConcater
    elif enc_type is PosEncType.LRRWP:
        preprocessor = RRWPPreprocessor
        concater = MLPConcater
    elif enc_type is PosEncType.DISTEMB:
        preprocessor = DistancePreprocessor
        concater = DistanceEmbeddingsConcater
    elif enc_type is PosEncType.LDISTEMB:
        preprocessor = DistancePreprocessor
        concater = MLPConcater
    else:
        raise ValueError("unsupported postional encodig type")
    
    preprocessor = preprocessor(positional_encodig_config, data_dir, is_inductive)
    concater = concater(positional_encodig_config)
    return preprocessor, concater


class PositionalEncoderPreprocessor(ABC):
    def __init__(self, positional_encodig_config, data_dir, is_inductive=False):
        super().__init__()
        
        self.positional_encodig_config = positional_encodig_config
        self.is_inductive = is_inductive
        self.force = False
        self.data_dir = data_dir
        if self.is_inductive:
            self.data_dir = None
            self.force = True
            
        self.model_emb = None
        self.model_builder = None
        
    def preprocess(self, g):
        raise NotImplementedError
    
    @abstractmethod
    def update(self, position_feature, g, obs_idx):
        raise NotImplementedError
    
    def __call__(self, student_g, g, obs_idx=None):
        pos = self.preprocess(student_g)
        if self.is_inductive:
            pos = self.update(pos, g, obs_idx)
        return pos


class DeepWalkPreprocessor(PositionalEncoderPreprocessor):
    def preprocess(self, g):
        self.model_emb = DeepWalkPE(g, data_dir=self.data_dir,
                    walk_length=self.positional_encodig_config["walk_length"], \
                    num_walks=self.positional_encodig_config["num_walks"],
                    window_size=self.positional_encodig_config["window_size"],
                    iter=self.positional_encodig_config["iter"], 
                    embed_size=self.positional_encodig_config["emb_size"],
                    workers=1)
        embeddings = self.model_emb.get_embeddings(force=self.force)    
        return embeddings
    
    def update(self, position_feature, g, obs_idx):
        updated_features = torch.zeros((g.num_nodes(),position_feature.shape[1]),dtype=position_feature.dtype,device=position_feature.device)
        updated_features[obs_idx] = position_feature
        
        with g.local_scope():
            g.ndata["emb"] = updated_features
            idx = g.filter_nodes(lambda x: torch.sum(x.data["emb"], dim=1)==0)
            g.pull(idx, dgl.function.copy_u('emb', 'm'), dgl.function.mean('m', 'emb'))

            return g.ndata["emb"]
        

class RRWPPreprocessor(PositionalEncoderPreprocessor):
    def preprocess(self, g):
        self.model_emb = RRWPPE(g, data_dir=self.data_dir, 
                            walk_length=self.positional_encodig_config["walk_length"])
        embeddings = self.model_emb.get_embeddings(force=self.force)
        return embeddings
    
    def update(self, position_feature, g, obs_idx):
        model_emb = RRWPPE(g, data_dir=None, walk_length=self.model_emb.walk_length)
        updated_features = model_emb.get_embeddings(force=True)
        updated_features[obs_idx] = position_feature
        return updated_features
        


class InverseDistanceAttentionPreprocessor(PositionalEncoderPreprocessor):
    def preprocess(self, g):
        self.model_emb = InverseDistanceAttentionPE(g, data_dir=self.data_dir, 
                            max_dist=self.positional_encodig_config["walk_length"])
        embeddings = self.model_emb.get_embeddings(force=self.force)
        return embeddings
    
    def update(self, position_feature, g, obs_idx):
        model_emb = InverseDistanceAttentionPE(g, data_dir=None,
                                max_dist=self.model_emb.max_dist)
        updated_features = model_emb.get_embeddings(force=True)
        updated_features[obs_idx] = position_feature
        return updated_features


class RandomWalkAttentionPreprocessor(PositionalEncoderPreprocessor):
    def preprocess(self, g):
        self.model_emb = RandomWalkAttentionPE(g, data_dir=self.data_dir, 
                            max_dist=self.positional_encodig_config["walk_length"],
                            num_walks=self.positional_encodig_config["num_walks"],
                            workers=1)
        embeddings = self.model_emb.get_embeddings(force=self.force)
        return embeddings
    
    def update(self, position_feature, g, obs_idx):
        model_emb = RandomWalkAttentionPE(g, data_dir=None,
                                max_dist=self.model_emb.max_dist,
                                num_walks=self.model_emb.num_walks, workers=1)
        updated_features = model_emb.get_embeddings(force=True)
        updated_features[obs_idx] = position_feature
        return updated_features


class DistancePreprocessor(PositionalEncoderPreprocessor):
    def preprocess(self, g):
        self.model_emb = DistancePE(g, data_dir=self.data_dir, 
                            max_dist=self.positional_encodig_config["walk_length"])
        embeddings = self.model_emb.get_embeddings(force=self.force)
        return embeddings
    
    def update(self, position_feature, g, obs_idx):
        model_emb = DistancePE(g, data_dir=self.data_dir, 
                            max_dist=self.model_emb.max_dist)
        updated_features = model_emb.get_embeddings(force=True)
        updated_features[obs_idx] = position_feature
        return updated_features


class BasePE(ABC):
    def __init__(self, graph, data_dir=None):
        self.graph = graph
        self.data_dir = data_dir
        self._embeddings = None
    
    def _save_embeddings(self):
        filename = self._filename()
        filename = os.path.join(self.data_dir, filename)
        os.makedirs(self.data_dir, exist_ok=True)
        torch.save({
            "embeddings": self._embeddings,
            "metadata": self._get_metadata(),
        }, filename)
    
    def _try_load_embeddings(self):
        if self.data_dir is None:
            return None
        try:
            filename = self._filename()
            filename = os.path.join(self.data_dir, filename)
            emb_file = torch.load(filename)
        except:
            return None
        metadata = emb_file["metadata"]
        if self._metadata_checker(metadata):
            return emb_file["embeddings"]
        return None
        
    def get_embeddings(self, force=False):
        if force:
            self._embeddings = None
        if self._embeddings is None:
            if not force:
                self._embeddings = self._try_load_embeddings()
                if self._embeddings is not None:
                    return self._embeddings

            self._compute_embeddings()
        
            if self.data_dir is not None:
                self._save_embeddings()
        return self._embeddings
    
    @abstractproperty
    def _filename(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def _get_metadata(self):
        raise NotImplementedError
    
    @abstractmethod
    def _metadata_checker(self, metadata) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def _compute_embeddings(self):
        raise NotImplementedError
    

class DeepWalkPE(BasePE):
    def __init__(self, graph, data_dir=None, walk_length=80, num_walks=10,
                embed_size=128, window_size=5, workers=1, iter=3):
        super().__init__(graph, data_dir)
        
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embed_size = embed_size
        self.window_size = window_size
        self.workers = workers
        self.iter = iter
        self.w2v_model = None
    
    def _filename(self) -> str:
        return f"deepwalk_{self.walk_length}_{self.num_walks}_{self.embed_size}_{self.window_size}_{self.iter}.pt"
    
    def _get_metadata(self):
        return  {
                    "walk_length": self.walk_length,
                    "num_walks": self.num_walks,
                    "embed_size": self.embed_size,
                    "window_size": self.window_size,
                    "iter": self.iter
                }
    
    def _metadata_checker(self, metadata) -> bool:
        return metadata["walk_length"] == self.walk_length and \
                    metadata["num_walks"] == self.num_walks and \
                    metadata["embed_size"] == self.embed_size and \
                    metadata["window_size"] == self.window_size and \
                    metadata["iter"] == self.iter
    
    def _compute_embeddings(self):
        graph = self.graph.cpu().to_networkx()
        walker = RandomWalker(graph, p=1, q=1)
        sentences = walker.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_length, workers=self.workers, verbose=0)

        kwargs = {}
        kwargs["sentences"] = sentences
        kwargs["min_count"] = 0
        kwargs["vector_size"] = self.embed_size  # vector_size, embed_size, 1433
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = self.workers
        kwargs["window"] = self.window_size
        kwargs["epochs"] = self.iter

        w2v_model = Word2Vec(**kwargs)
        
        embeddings = []
        for word in graph.nodes():
            embeddings.append(w2v_model.wv[word])

        embeddings = np.array([w2v_model.wv[word] for word in graph.nodes()], dtype=np.float32)
        self._embeddings = torch.from_numpy(embeddings)
        

class RRWPPE(BasePE):
    def __init__(self, graph, data_dir=None, walk_length=18):
        super().__init__(graph, data_dir)
        
        self.walk_length = walk_length
        
    def _filename(self) -> str:
        return  f"rrwpe_{self.walk_length}.pt"
    
    def _get_metadata(self):
        return  {
                    "walk_length": self.walk_length,
                }
    
    def _metadata_checker(self, metadata) -> bool:
        return metadata["walk_length"] == self.walk_length
    
    def _compute_embeddings(self):
        adj = self.graph.cpu().adj()
        num_nodes = self.graph.num_nodes()
        deg = 1.0 / adj.sum(dim=1)
        deg[deg == float('inf')] = 0
        adj = adj * dgl.sparse.diag(deg)
        adj = adj.to_dense()
        
        # add identity
        pe_list = [
            torch.ones((num_nodes,), dtype=adj.dtype, device=adj.device),
            adj.diagonal()
        ]

        out = adj
        if self.walk_length > 2:
            for _ in range(2, self.walk_length):
                out = out @ adj
                pe_list.append(out.diagonal())
        
        self._embeddings = torch.stack(pe_list, dim=-1) # NxK



class DistancePE(BasePE):
    def __init__(self, graph, data_dir=None, max_dist=5):
        super().__init__(graph, data_dir)
        
        self.max_dist = max_dist
        
    def _filename(self) -> str:
        return  f"int_dist_att_{self.max_dist}.pt"
    
    def _get_metadata(self):
        return  {
                    "max_dist": self.max_dist,
                }
    
    def _metadata_checker(self, metadata) -> bool:
        return metadata["max_dist"] == self.max_dist
    
    def _compute_embeddings(self):
        graph = self.graph.cpu().to_networkx()
        
        att = torch.zeros((graph.number_of_nodes(),graph.number_of_nodes()))
        for n in graph:
            for s, l in nx.single_target_shortest_path_length(graph, n, cutoff=self.max_dist):
                att[n, s] = l
        
        return att



class AttentionPE(BasePE):   
    @abstractmethod
    def _get_attention_mat(self):
        raise NotImplementedError
    
    def _compute_embeddings(self):
        att = self._get_attention_mat()
        att = torch.softmax(att,dim=1)
        self._embeddings = att @ self.graph.ndata["feat"]



class InverseDistanceAttentionPE(AttentionPE):
    def __init__(self, graph, data_dir=None, max_dist=5):
        super().__init__(graph, data_dir)
        
        self.max_dist = max_dist
        
    def _filename(self) -> str:
        return  f"int_dist_att_{self.max_dist}.pt"
    
    def _get_metadata(self):
        return  {
                    "max_dist": self.max_dist,
                }
    
    def _metadata_checker(self, metadata) -> bool:
        return metadata["max_dist"] == self.max_dist
    
    def _get_attention_mat(self):
        graph = self.graph.cpu().to_networkx()
        
        att = torch.zeros((graph.number_of_nodes(),graph.number_of_nodes()))
        for n in graph:
            for s, l in nx.single_target_shortest_path_length(graph, n, cutoff=self.max_dist):
                att[n, s] = l
        
        return 1/(att + 1)


class RandomWalkAttentionPE(AttentionPE):
    def __init__(self, graph, data_dir=None, walk_length=80, num_walks=10, workers=1):
        super().__init__(graph, data_dir)
        
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        
    def _filename(self) -> str:
        return  f"rw_att{self.max_dist}.pt"
    
    def _get_metadata(self):
        return  {
                    "walk_length": self.walk_length,
                    "num_walks": self.num_walks
                }
    
    def _metadata_checker(self, metadata) -> bool:
        return metadata["walk_length"] == self.walk_length and \
                    metadata["num_walks"] == self.num_walks
    
    def _get_attention_mat(self):
        graph = self.graph.cpu().to_networkx()
        walker = RandomWalker(graph, p=1, q=1)
        sentences = walker.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_length, workers=self.workers, verbose=0)
        
        att = torch.zeros((graph.number_of_nodes(),graph.number_of_nodes()))
        for s in sentences:
            start = s[0]
            for f, n in  Counter(s[1:]).items():
                att[start,f] +=n
        att /= self.num_walks
        att /= self.walk_length
        return att


class EmbeddingsConcater(torch.nn.Module):
    def __init__(self, positional_encodig_config):
        super().__init__()
    
    def process(self, enc):
        return enc
    
    def forward(self, features, encodings):
        enc = self.process(encodings)
        return torch.cat([features, enc], dim=1)


class MLPConcater(EmbeddingsConcater):
    def __init__(self, positional_encodig_config):
        super().__init__()
        
        self.emb = None
        emb_dim = positional_encodig_config.get(emb_dim, None)
        out_dim = positional_encodig_config.get(out_dim, None)
        vocab_size = positional_encodig_config.get(vocab_size, None)
        if vocab_size is not None:
            self.emb = torch.nn.Embedding(vocab_size, emb_dim)
        self.fc = torch.nn.Linear(emb_dim, out_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)
    
    def process(self, enc):
        out = enc
        if self.emb is not None:
            out = self.emb(out)
        out = self.fc(out)
        return out



class DistanceEmbeddingsConcater(EmbeddingsConcater):
    def __init__(self, positional_encodig_config):
        super().__init__()
        
        self.emb = None
        out_dim = positional_encodig_config.get(out_dim, None)
        vocab_size = positional_encodig_config.get(vocab_size, None)
        self.emb = torch.nn.Embedding(vocab_size, out_dim)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
    
    def process(self, enc):
        return self.emb(enc)
