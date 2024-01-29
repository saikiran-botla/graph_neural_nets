# %%
from sgmatch.models.GMN import GMNEmbed,GMNMatch
import torch
from torch_geometric.utils.random import erdos_renyi_graph
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import tqdm
from tests.utils.data import PairData
import random
import pickle

# %%
class TripletData(Data):
    def __init__(self, edge_index_1=None, x_1=None, 
                edge_index_2=None, x_2=None, edge_index_3=None,x_3=None,y=None):
        super(TripletData, self).__init__()  # Call the parent class constructor here
        self.edge_index_1 = edge_index_1
        self.x_1 = x_1
        self.edge_index_2 = edge_index_2
        self.x_2 = x_2
        self.edge_index_3 = edge_index_3
        self.x_3 = x_3
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_1':
            return self.x_1.size(0)
        if key == 'edge_index_2':
            return self.x_2.size(0)
        if key == 'edge_index_3':
            return self.x_3.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
    def __repr__(self):
        return '{}(x_1 = {}, edge_index_1 = {}, x_2 = {}, edge_index_2 = {}, x_3 = {}, edge_index_3 = {})'.format(
            self.__class__.__name__, self.x_1.shape, self.edge_index_1.shape,
            self.x_2.shape, self.edge_index_2.shape, self.x_3.shape, self.edge_index_3.shape
        )


# %%
class PairData(Data):
    r"""
    """
    def __init__(self, edge_index_s=None, x_s=None, 
                edge_index_t=None, x_t=None, y=None):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __repr__(self):
        return '{}(x_s = {}, edge_index_s = {}, x_t = {}, edge_index_t = {}, y = {})'.format(
            self.__class__.__name__, self.x_s.shape, self.edge_index_s.shape,
            self.x_t.shape, self.edge_index_t.shape, self.y.shape
        )

# %%

def create_graphs(num_nodes,edge_probability,size,kp=1,kn=2):
    G1=[] #Random binomial drawn graphs
    G2=[] #Graphs obtained by replacing kp edges to find positive graphs to G1
    G3=[] #Graphs obtained by replacing kn edges to find negative graphs to G2

    with tqdm.tqdm(total=size,desc="Creating Graphs:") as bar:
        for _ in range(size):
            e1=erdos_renyi_graph(num_nodes,edge_probability)

            id=torch.randint(0,e1.shape[1],(kp,))
            indices1=[e for e in range(e1.shape[1]) if e not in id]
            
            e2=e1[:,indices1]

            id=torch.randint(0,e1.shape[1],(kn,))
            indices2=[e for e in range(e1.shape[1]) if e not in id]

            e3=e1[:,indices2]

            data=Data(x=torch.ones(num_nodes,32),edge_index=e1,num_nodes=num_nodes)
            data1=Data(x=torch.ones(num_nodes,32),edge_index=e2,num_nodes=num_nodes)
            data2=Data(x=torch.ones(num_nodes,32),edge_index=e3,num_nodes=num_nodes)

            G1.append(data)
            G2.append(data1)
            G3.append(data2)

            bar.update(1)
    return G1,G2,G3


# %%
class GMN_loss(torch.nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(GMN_loss, self).__init__()

    def forward(self, g1_pred,g2_pred,g3_pred,gamma):
        loss=torch.sqrt(torch.sum(torch.pow(torch.subtract(g1_pred.reshape((g1_pred.shape[-1],)), g2_pred.reshape((g1_pred.shape[-1],))), 2), dim=0))-torch.sqrt(torch.sum(torch.pow(torch.subtract(g1_pred.reshape((g1_pred.shape[-1],)),g3_pred.reshape((g1_pred.shape[-1],))), 2), dim=0))+gamma
        #print(loss.shape)
        loss=torch.maximum(torch.tensor(0),loss)
        return loss

# %%
def train(g1_train, g1_val,g2_train, g2_val,g3_train, g3_val, model, loss_criterion, optimizer, device, num_epochs=10, gamma=0.1):

    for epoch in range(num_epochs):
        train_loss_sum = 0
        val_loss_sum = 0
        with tqdm.tqdm(total=len(g1_train)*len(g1_train[0]), desc='Train batches completed: ') as bar:
            for i in range(len(g1_train)):
                model.train()
                optimizer.zero_grad()
                batch_loss=0
                for j in range(len(g1_train[0])):
                    
                    g1_pred = model(g1_train[i][j].x, g1_train[i][j].edge_index)
                    g2_pred = model(g2_train[i][j].x, g2_train[i][j].edge_index)
                    g3_pred = model(g3_train[i][j].x, g3_train[i][j].edge_index)
                    
                    batch_loss += loss_criterion(g1_pred,g2_pred,g3_pred,gamma)
                    # Compute Gradients via Backpropagation
                
                train_loss_sum+=batch_loss
                batch_loss.backward()
                optimizer.step()
                
                bar.update(len(g1_train[0]))

        with tqdm.tqdm(total=len(g1_val)*len(g1_val[0]), desc='Validation batches completed: ') as bar:
            for i in range(len(g1_val)):
                model.eval()
                batch_loss=0
                for j in range(len(g1_val[0])):
                    with torch.no_grad():
                        g1_pred = model(g1_val[i][j].x, g1_val[i][j].edge_index)
                        g2_pred = model(g2_val[i][j].x, g2_val[i][j].edge_index)
                        g3_pred = model(g3_val[i][j].x, g3_val[i][j].edge_index)
                        
                        batch_loss += loss_criterion(g1_pred,g2_pred,g3_pred,gamma)
                val_loss_sum+=batch_loss
                bar.update(len(g1_val[0]))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
    
        # Printing Epoch Summary
        print(f"Epoch: {epoch+1}/{num_epochs} | Per Graph Train MSE: {train_loss_sum / (len(g1_train)*len(g1_train[0]))} | Per Graph Validation MSE: {val_loss_sum / (len(g1_val)*len(g1_val[0]))}")

# %%
G1,G2,G3=create_graphs(20,0.3,100000)

# %%
def divide_batchwise(dataset,batch_size):
    l=len(dataset)
    batched_dataset=[]
    prev=0
    for i in range(0,l,batch_size):
        if i==0:
            prev=i
            continue
        if i<l:
            batched_dataset.append(dataset[prev:i])
            prev=i
        else:
            batched_dataset.append(dataset[prev:])

    return batched_dataset

# %%


# %%
t1=int(0.65*len(G1))
t2=int(0.75*len(G1))
t1=10000
t2=15000
t3=20000

g1_train=divide_batchwise(G1[:t1],128)
g1_val=divide_batchwise(G1[t1+1:t2],64)
g1_test=divide_batchwise(G1[t2+1:t3],256)

g2_train=divide_batchwise(G2[:t1],128)
g2_val=divide_batchwise(G2[t1+1:t2],64)
g2_test=divide_batchwise(G2[t2+1:t3],256)

g3_train=divide_batchwise(G3[:t1],128)
g3_val=divide_batchwise(G3[t1+1:t2],64)
g3_test=divide_batchwise(G3[t2+1:t3],256)


D=32
learning_rate=0.01

mlp_layers=[32,16,8]

model = GMNEmbed(node_feature_dim=g1_train[0][0].x.shape[-1], 
                enc_node_hidden_sizes=mlp_layers,
                prop_node_hidden_sizes=mlp_layers,
                prop_message_hidden_sizes=mlp_layers,
                aggr_gate_hidden_sizes=mlp_layers,
                aggr_mlp_hidden_sizes=mlp_layers)

criterion=GMN_loss()
optimizer=torch.optim.Adam(model.parameters(),learning_rate)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



# %%
g1_pred=torch.Tensor([1,2,3,4])
g2_pred=torch.Tensor([2,3,4,5])

#g1_pred.reshape((1,4))
torch.sum(torch.pow(torch.subtract(g1_pred, g2_pred), 2), dim=0)

# %%
train(g1_train, g1_val,g2_train, g2_val,g3_train, g3_val, model, criterion, optimizer, device, num_epochs=10, gamma=0.1)

# %%
def eculdiean_dist(a,b):
    return torch.sqrt(torch.sum(torch.pow(torch.subtract(a.reshape((a.shape[-1],)), b.reshape((a.shape[-1],))), 2), dim=0))

# %%
accuracy=0

with tqdm.tqdm(total=len(g1_test)*len(g1_test[0]),desc="testing: ") as bar:
    for i in range(len(g1_test)):
        for j in range(len(g1_test[0])):
            g1_pred = model(g1_test[i][j].x, g1_test[i][j].edge_index)
            g2_pred = model(g2_test[i][j].x, g2_test[i][j].edge_index)
            g3_pred = model(g3_test[i][j].x, g3_test[i][j].edge_index)

            if eculdiean_dist(g1_pred,g2_pred) < eculdiean_dist(g1_pred,g3_pred) :
                accuracy+=1
        bar.update(len(g1_test[0]))

accuracy=accuracy/(len(g1_test)*len(g1_test[0]))
accuracy



