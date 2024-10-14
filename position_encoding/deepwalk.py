import os
import numpy as np
import torch
from .walker import RandomWalker
from gensim.models import Word2Vec


class DeepWalk(object):
    def __init__(self, graph, data_dir=None, walk_length=80, num_walks=10):

        self.graph = graph
        self.data_dir = data_dir
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.w2v_model = None
        self._embeddings = None

    def train(self, embed_size=128, window_size=5, workers=1, iter=3, **kwargs):
        graph = self.graph.cpu().to_networkx()
        walker = RandomWalker(graph, p=1, q=1)
        sentences = walker.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_length, workers=workers, verbose=0)
        
        kwargs["sentences"] = sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size  # vector_size, embed_size, 1433
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        model = Word2Vec(**kwargs)

        self.w2v_model = model
        return model
    
    def _get_save_filepath(self, embed_size, window_size, iter):
        filename = self.data_dir
        if not os.path.isfile(filename):
            os.makedirs(filename, exist_ok=True)
            filename = f"deepwalk_{self.walk_length}_{self.num_walks}_{embed_size}_{window_size}_{iter}.pt"
            filename = os.path.join(self.data_dir, filename)
        return filename
    
    def _try_load_embeddings(self, embed_size, window_size, iter):
        if self.data_dir is None:
            return None
        filename = self._get_save_filepath(embed_size, window_size, iter)
        try:
            emb_file = torch.load(filename)
        except:
            return None
        metadata = emb_file["metadata"]
        if  metadata["walk_length"] == self.walk_length and \
                metadata["num_walks"] == self.num_walks and \
                metadata["embed_size"] == embed_size and \
                metadata["window_size"] == window_size and \
                metadata["iter"] == iter:
            return emb_file["embeddings"]
        return None

    def get_embeddings(self, embed_size=128, window_size=5, workers=1, iter=3, force=False, **kwargs):
        if force:
            self._embeddings = None
        if self._embeddings is None:
            if not force:
                self._embeddings = self._try_load_embeddings(embed_size, window_size, iter)
                if self._embeddings is not None:
                    return self._embeddings

            graph = self.graph.cpu().to_networkx()
            walker = RandomWalker(graph, p=1, q=1)
            sentences = walker.simulate_walks(num_walks=self.num_walks, walk_length=self.walk_length, workers=workers, verbose=0)

            kwargs["sentences"] = sentences
            kwargs["min_count"] = kwargs.get("min_count", 0)
            kwargs["vector_size"] = embed_size  # vector_size, embed_size, 1433
            kwargs["sg"] = 1  # skip gram
            kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
            kwargs["workers"] = workers
            kwargs["window"] = window_size
            kwargs["epochs"] = iter

            w2v_model = Word2Vec(**kwargs)
            
            embeddings = []
            for word in graph.nodes():
                embeddings.append(w2v_model.wv[word])

            embeddings = np.array([w2v_model.wv[word] for word in graph.nodes()], dtype=np.float32)
            self._embeddings = torch.from_numpy(embeddings)
            
            if self.data_dir is not None:
                filename = self._get_save_filepath(embed_size, window_size, iter)
                torch.save({
                    "embeddings": self._embeddings,
                    "metadata": {
                        "walk_length": self.walk_length,
                        "num_walks": self.num_walks,
                        "embed_size": embed_size,
                        "window_size": window_size,
                        "iter": iter
                    }
                }, filename)
        return self._embeddings
