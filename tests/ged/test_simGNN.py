from typing import List, Tuple
import tqdm
import os
import os.path as osp

import torch
from torch.functional import Tensor
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.utils import to_dense_batch

from sgmatch.models.SimGNN import SimGNN
from tests.utils.dataset import load_dataset
from tests.utils.parser import parser
from tests.utils.data import PairData

def create_graph_pairs(train_dataset, test_dataset) -> Tuple[List]:
    train_graph_pairs = []
    with tqdm.tqdm(total=len(train_dataset)**2, desc='Train graph pairs completed: ') as bar:
        for idx1, graph1 in enumerate(train_dataset):
            for idx2, graph2 in enumerate(train_dataset):
                if idx1 == idx2:
                    continue
                # Initializing Data
                edge_index_s = graph1.edge_index
                x_s = graph1.x

                edge_index_t = graph2.edge_index
                x_t = graph2.x

                norm_ged = train_dataset.norm_ged[graph1.i, graph2.i]
                graph_sim = torch.exp(-norm_ged).unsqueeze(-1)
                
                # Making Graph Pair
                if isinstance(x_s, Tensor) and isinstance(x_t, Tensor):
                    graph_pair = PairData(edge_index_s=edge_index_s, x_s=x_s,
                                        edge_index_t=edge_index_t, x_t=x_t,
                                        y=graph_sim)
                    
                    # Saving all the Graph Pairs to the List for Batching and Data Loading
                    train_graph_pairs.append(graph_pair)
            bar.update(len(train_dataset))
    
    test_graph_pairs = []
    with tqdm.tqdm(total=len(test_dataset)*len(train_dataset), desc='Test graph pairs completed: ') as bar:
        for graph1 in test_dataset:
            for graph2 in train_dataset:
                # Initializing Data
                edge_index_s = graph1.edge_index
                x_s = graph1.x
                edge_index_t = graph2.edge_index
                x_t = graph2.x

                norm_ged = train_dataset.norm_ged[graph1.i, graph2.i]
                graph_sim = torch.exp(-norm_ged).unsqueeze(-1)
                
                # Making Graph Pair
                if isinstance(x_s, Tensor) and isinstance(x_t, Tensor):
                    graph_pair = PairData(edge_index_s=edge_index_s, x_s=x_s,
                                        edge_index_t=edge_index_t, x_t=x_t,
                                        y=graph_sim)
                
                # Saving all the Graph Pairs to the List for Batching and Data Loading
                test_graph_pairs.append(graph_pair)
            bar.update(len(train_dataset))
    
    return train_graph_pairs, test_graph_pairs

def train(train_loader, val_loader, model, loss_criterion, optimizer, device, num_epochs=10):
    batch_train_loss_sum = 0
    batch_val_loss_sum = 0

    for epoch in range(num_epochs):
        with tqdm.tqdm(total=len(train_loader), desc='Train batches completed: ') as bar:
            for batch_idx, train_batch in enumerate(train_loader):
                model.train()
                train_batch = train_batch.to(device)
                optimizer.zero_grad()

                pred_sim = model(train_batch.x_s, train_batch.edge_index_s, train_batch.x_t, 
                                train_batch.edge_index_t, train_batch.x_s_batch, train_batch.x_t_batch)
                mean_batch_loss = loss_criterion(pred_sim, train_batch.y)
                # Compute Gradients via Backpropagation
                mean_batch_loss.backward()
                # Update Parameters
                optimizer.step()
                batch_train_loss_sum += mean_batch_loss.item()*len(train_batch)
                
                bar.update(1)

        with tqdm.tqdm(total=len(val_loader), desc='Validation batches completed: ') as bar:
            for batch_idx, val_batch in enumerate(val_loader):
                model.eval()
                with torch.no_grad():
                    val_batch = val_batch.to(device)
                    pred_sim = model(val_batch.x_s, val_batch.edge_index_s, 
                            val_batch.x_t, val_batch.edge_index_t, val_batch.x_s_batch, val_batch.x_t_batch)
                    mean_val_loss = loss_criterion(pred_sim, val_batch.y)
                    batch_val_loss_sum += mean_val_loss.item()*len(val_batch)

                bar.update(1)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
    
        # Printing Epoch Summary
        print(f"Epoch: {epoch+1}/{num_epochs} | Per Graph Train MSE: {batch_train_loss_sum / len(train_loader.dataset)} | Per Graph Validation MSE: {batch_val_loss_sum / len(val_loader.dataset)}")

if __name__ == "__main__":
    # Run this from root folder
    # python -m tests.ged.test_aids --data_path <path_to_data_folder> --create_graph_pairs

    args = parser.parse_args()
    print(args.train_batch_size, args.val_batch_size,args.test_batch_size,args.learning_rate)
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'

    train_dataset = load_dataset(dpath=args.data_path+"/aids/", name="GED", category="AIDS700nef", train=True)
    test_dataset = load_dataset(dpath=args.data_path+"/aids/", name="GED", category="AIDS700nef", train=False)

    train_ged_table = train_dataset.ged[:train_dataset.data.i[-1]+1, :train_dataset.data.i[-1]+1]
    test_ged_table = test_dataset.ged[train_dataset.data.i[-1]+1:, train_dataset.data.i[-1]+1:]

    if args.create_graph_pairs:
        train_graph_pairs, test_graph_pairs = create_graph_pairs(train_dataset, test_dataset)
        if not osp.exists(args.data_path+"/aids/graph_pairs"):
            os.makedirs(args.data_path+"/aids/graph_pairs")
        torch.save(train_graph_pairs, args.data_path+"/aids/graph_pairs/train_graph_pairs.pt")
        torch.save(test_graph_pairs, args.data_path+"/aids/graph_pairs/test_graph_pairs.pt")
    else:
        train_graph_pairs, test_graph_pairs = torch.load(args.data_path+"/aids/graph_pairs/train_graph_pairs.pt"),\
                                              torch.load(args.data_path+"/aids/graph_pairs/test_graph_pairs.pt")
    
    val_idxs = np.random.randint(len(train_graph_pairs), size=len(test_graph_pairs))
    val_graph_pairs = [train_graph_pairs[idx] for idx in val_idxs]
    train_idxs = set(range(len(train_graph_pairs))) - set(val_idxs)
    train_graph_pairs = [train_graph_pairs[idx] for idx in train_idxs]
    del val_idxs, train_idxs

    

    train_loader = DataLoader(train_graph_pairs, batch_size = args.train_batch_size, follow_batch = ["x_s", "x_t"], shuffle = True)
    val_loader = DataLoader(val_graph_pairs, batch_size = args.val_batch_size, follow_batch = ["x_s", "x_t"], shuffle = True)
    test_loader = DataLoader(test_graph_pairs, batch_size = args.test_batch_size, follow_batch = ["x_s", "x_t"], shuffle = True)

    
    model = SimGNN(input_dim=train_loader.dataset[0].x_s.shape[-1]).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    train(train_loader, val_loader, model, criterion, optimizer, device)


    
    