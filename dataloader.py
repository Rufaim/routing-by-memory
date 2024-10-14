import os
import enum
import numpy as np
import dgl
import dgl.data
import torch
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import scipy.sparse as sp
from ogb.nodeproppred import DglNodePropPredDataset


class DataType(enum.Enum):
    CORA = "cora"
    CITESEER = "citeseer"
    PUBMED = "pubmed"
    AMAZON_COM = "amazon-com"
    AMAZON_PHOTO = "amazon-photo"
    ACADEMIC_CS = "academic-cs"
    ACADEMIC_PHYSICS = "academic-physics"
    OGBN_ARXIV = "ogbn-arxiv"
    OGBN_PRODUCTS = "ogbn-products"

    def __str__(self):
        return self.value


def load_data(datatype: DataType, dataset_path: str, seed=None, labelrate_train=20, labelrate_val=30, verbose=True):
    if datatype in [DataType.CORA, DataType.CITESEER, DataType.PUBMED]:
        return load_dgl_citations_data(datatype, dataset_path, verbose)
    if datatype in [DataType.AMAZON_COM, DataType.AMAZON_PHOTO,
                    DataType.ACADEMIC_CS, DataType.ACADEMIC_PHYSICS]:
        return load_cpf_data(datatype, dataset_path, seed, labelrate_train, labelrate_val)
    if datatype in [DataType.OGBN_ARXIV, DataType.OGBN_PRODUCTS]:
        return load_ogb_data(datatype, dataset_path)
    else:
        raise ValueError(f"Unknown dataset type: {datatype}")


def load_dgl_citations_data(datatype, dataset_path, verbose):
    if datatype is DataType.CORA:
        graph = dgl.data.CoraGraphDataset(raw_dir=dataset_path, verbose=verbose)[0]
    elif datatype is DataType.CITESEER:
        graph = dgl.data.CiteseerGraphDataset(raw_dir=dataset_path, verbose=verbose)[0]
    elif datatype  is DataType.PUBMED:
        graph = dgl.data.PubmedGraphDataset(raw_dir=dataset_path, verbose=verbose)[0]
    else:
        raise ValueError(f"Unknown dgl dataset type: {datatype}")

    labels = graph.ndata['label']
    graph = dgl.add_self_loop(graph)
    
    train_mask = graph.ndata['train_mask']
    test_mask = graph.ndata['test_mask']
    val_mask = graph.ndata['val_mask']

    return graph, labels, train_mask.nonzero()[:, 0], val_mask.nonzero()[:, 0], test_mask.nonzero()[:, 0]


def load_cpf_data(datatype, dataset_path, seed, labelrate_train=20, labelrate_val=30):
    if datatype is DataType.AMAZON_COM:
        datapath = os.path.join(dataset_path, "amazon_electronics_computers.npz")
    elif datatype is DataType.AMAZON_PHOTO:
        datapath = os.path.join(dataset_path, "amazon_electronics_photo.npz")
    elif datatype is DataType.ACADEMIC_CS:
        datapath = os.path.join(dataset_path, "ms_academic_cs.npz")
    elif datatype is DataType.ACADEMIC_PHYSICS:
        datapath = os.path.join(dataset_path, "ms_academic_phy.npz")
    else:
        raise ValueError(f"Unknown cpf dataset type: {datatype}")
    
    if not os.path.isfile(datapath):
        raise RuntimeError(f"{datapath} doesn't exist.")

    adj_matrix, features, labels = load_cpf_from_npz(datapath)

    # preprocess graph
    adj_matrix = standartize_cpf_graph(adj_matrix)
    adj_matrix, features, labels = largest_connected_component(adj_matrix, features, labels)
    labels = binarize_labels(labels)

    splitter = TrainValTestSplitter(labelrate_train, labelrate_val, seed)
    idx_train, idx_val, idx_test = splitter(labels)

    features = torch.from_numpy(np.array(features.todense(), dtype=np.float32))
    labels = torch.from_numpy(labels.argmax(axis=1).astype(np.int64))

    adj = normalize_adj(adj_matrix)
    adj_sp = adj_matrix.tocoo()
    g = dgl.graph((adj_sp.row, adj_sp.col))
    g.ndata["feat"] = features

    idx_train = torch.from_numpy(idx_train.astype(np.int64))
    idx_val = torch.from_numpy(idx_val.astype(np.int64))
    idx_test = torch.from_numpy(idx_test.astype(np.int64))

    return g, labels, idx_train, idx_val, idx_test


def binarize_labels(labels, sparse_output=False, return_classes=False):
    """Convert labels vector to a binary label matrix.

    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].

    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.

    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.

    """
    if hasattr(labels[0], "__iter__"):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def normalize_adj(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    adj_ = r_mat_inv.dot(adj_)
    return adj_


def load_cpf_from_npz(file_name):
    with np.load(file_name, allow_pickle=True) as loader:
        loader = loader
        adj_matrix = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )
        attr_matrix = sp.csr_matrix(
                (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
                shape=loader["attr_shape"],
            )
        labels = loader["labels"]

    return adj_matrix, attr_matrix, labels


def standartize_cpf_graph(adj):
    """unweighted/undirected/no-self-loop graph"""
    # unweigth
    adj.data = np.ones_like(adj.data) 
    
    # undirect
    adj = adj + adj.T
    adj[adj != 0] = 1
    
    return eliminate_self_loops_sparse_adj(adj)
    

def eliminate_self_loops_sparse_adj(A):
    """Remove self-loops from the sparse adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def largest_connected_component(adj_matrix, features, labels):
    _, component_indices = sp.csgraph.connected_components(adj_matrix)
    component_sizes = np.bincount(component_indices)
    largest_componet_idx = np.argmax(component_sizes)
    nodes_to_keep = sorted([
        idx
        for (idx, component) in enumerate(component_indices)
        if component == largest_componet_idx
    ])
    
    adj_matrix = adj_matrix[nodes_to_keep][:, nodes_to_keep]
    features = features[nodes_to_keep]
    labels = labels[nodes_to_keep]
    return adj_matrix, features, labels



class TrainValTestSplitter(object):
    def __init__(self, train_examples_per_class, val_examples_per_class, seed):
        self.train_examples_per_class = train_examples_per_class
        self.val_examples_per_class = val_examples_per_class
        self.random_gen = np.random.default_rng(seed)
        
    def __call__(self, labels):
        num_samples, _ = labels.shape
        train_indices = self.sample_per_class(labels, self.train_examples_per_class)
        val_indices = self.sample_per_class(labels, self.val_examples_per_class, 
                                    forbidden_indices=train_indices)
        
        remaining_indices = list(range(num_samples))
        forbidden_indices = np.concatenate((train_indices, val_indices))
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        
        # assert that there are no duplicates in sets
        assert len(set(train_indices)) == len(train_indices)
        assert len(set(val_indices)) == len(val_indices)
        assert len(set(test_indices)) == len(test_indices)
        # assert sets are mutually exclusive
        assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
        assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
        assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
        # all indices must be part of the split
        assert (train_indices.shape[0] + val_indices.shape[0] + test_indices.shape[0]) == num_samples 
        # assert all classes have equal cardinality
        assert np.unique(np.sum(labels[train_indices, :], axis=0)).size == 1
        assert np.unique(np.sum(labels[val_indices, :], axis=0)).size == 1

        return train_indices, val_indices, test_indices
    
    def sample_per_class(self, labels, num_examples_per_class, forbidden_indices=None):
        num_samples, num_classes = labels.shape
        sample_indices_per_class = {index: [] for index in range(num_classes)}
        
        # get indices sorted by class
        for class_index in range(num_classes):
            for sample_index in range(num_samples):
                if labels[sample_index, class_index] > 0.0:
                    if forbidden_indices is None or sample_index not in forbidden_indices:
                        sample_indices_per_class[class_index].append(sample_index)
    
        return np.concatenate([
                self.random_gen.choice(
                    sample_indices_per_class[class_index],
                    num_examples_per_class,
                    replace=False,
                )
                for class_index in range(len(sample_indices_per_class))
            ])


def load_ogb_data(datatype, dataset_path):
    data = DglNodePropPredDataset(datatype.value, dataset_path)
    splitted_idx = data.get_idx_split()
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    g, labels = data[0]
    labels = labels.squeeze()

    if datatype is DataType.OGBN_ARXIV:
        # to undirect graph
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
    g = g.add_self_loop()

    return g, labels, idx_train, idx_val, idx_test

