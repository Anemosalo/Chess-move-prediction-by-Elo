import torch
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import time
#import matplotlib.pyplot as plt

def load_shard(path):
    d = np.load(path)
    return tuple(d[k] for k in ("planes","y","legal_flat","offsets","rating","split"))


def get_batch(idx,planes , y ,legal_flat ,offsets, rating):
    #-------------#
    # Output 1: X -> Tensor of [B,18,8,8]
    B = len(idx) #Batch Size
    X = torch.from_numpy(planes[idx]).float() ##Tensor of Dim (B,17,8,8)
    ratings = (torch.from_numpy(rating[idx]).float() -1500)/400 ##Tensor [B]
    ratings = ratings.reshape(B,1,1,1).expand(B,1,8,8) ## Tensor [B,1,8,8]
    X = torch.cat([X, ratings ],dim=1) # Tensor of Dim (B,18,8,8)
    #---------------#

    #--------------#
    # Output 2: Mask of legal moves -> Tensor [B,4168]#
    mask = torch.full((B,4168),float("-inf")) #Shape [B,4168] with -inf in every cell
    for k,j in enumerate(idx):
        legal = legal_flat[offsets[j]:offsets[j+1]]
        mask[k,legal]=0

    #--------------#
    #Output 3: the played move -> Tensor [B] 
    y = torch.from_numpy(y[idx]).long()

    #-------------#

    return X, mask, y
    

class ResBlock(nn.Module): # One residual block of input (B,64,8,8) and output (B,64,8,8)
    
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(64,64,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self. conv2 = nn.Conv2d(64,64,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self,x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out+x)


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.start_conv = nn.Conv2d(18,64,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.blocks = nn.Sequential(*[ResBlock() for _ in range(6)])
        self.exit_head = nn.Conv2d(64,73,1,bias=True)
    def forward(self,x):
        B = x.shape[0]
        X = torch.relu(self.bn1(self.start_conv(x))) #x (B,18,8,8) -> Conv1 -> (B,64,8,8) BatchNorm (B,64,8,8) -> relu -> (B,64,8,8)
        X = self.blocks(X) # In (B,64,8,8) -> Out (B,64,8,8) after 6 residual 3x3 layers block
        head = self.exit_head(X) # In (B,64,8,8) -> Out (B,73,8,8) 1x1 conv
        normal_moves = torch.permute(head[:,:64],(0,2,3,1) ).reshape(B,4096)
        underpromotions = torch.permute(head[:,64:,6,:],(0,2,1)).reshape(B,72)
        return torch.cat([normal_moves,underpromotions],dim=1)

def eval(model,rows,planes,y,legal_flat,offsets,rating,device):
    model.eval()
    total_bits= 0.0
    total_n = 0 #total positions
    with torch.no_grad():
        for start in range(0,len(rows),1024):
            idx = rows[start:start+1024] #next chunk of row nums
            X, mask , targets = get_batch(idx,planes,y,legal_flat,offsets,rating)
            X,mask,targets = X.to(device),mask.to(device),targets.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits+mask,targets)/math.log(2)
            total_bits+=loss.item()*len(idx)
            total_n+=len(idx)
        model.train()
        return total_bits/total_n

def main():
    device = "cuda"if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu") 
    ## Getting the Data##
    B = 1024
    planes, y, legal_flat, offsets, rating, split = load_shard("data/games_encoded/lichess_2026-09_elo1000-1199_blitz_000.npz")
    train_rows = np.flatnonzero(split == 0)
    val_rows = np.flatnonzero(split==1)
    idx = np.random.choice(train_rows, B, replace=False)
    X, mask, target = get_batch(idx, planes, y, legal_flat, offsets, rating)
    X,mask,target = X.to(device) , mask.to(device), target.to(device)
    ## NETWORK: ##

    
    model = Network().to(device)
    print("Before Training: ",eval(model,val_rows,planes,y,legal_flat,offsets,rating,device))
    model.train()
    optimizier = torch.optim.Adam(model.parameters(),lr=1e-3)
    best = float("inf")
    for epoch in range(3):
        t = time.time()
        order = np.random.permutation(train_rows)
        total_bits, total_n = 0.0,0
        for start in range(0,len(order),B):
            idx = order[start:start+B]
            X, mask, target = get_batch(idx, planes, y, legal_flat, offsets, rating)
            X,mask,target = X.to(device) , mask.to(device), target.to(device)
            logits = model(X)  
            loss = F.cross_entropy(logits+mask,target)/math.log(2) #calculated with natural log so we divide by log(2) to get bits meaning log_2 calculation instead of log_e
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
            total_bits = loss.item()*len(idx)
            total_n = len(idx)
        train_bits = total_bits/total_n
        val_bits = eval(model, val_rows, planes, y, legal_flat, offsets, rating, device)
        print(f"epoch {epoch}: train {train_bits:.3f}  val {val_bits:.3f}  ({time.time() - t:.0f}s)")
        if val_bits < best:
            best = val_bits
            torch.save(model.state_dict,"best.pt")


    





if __name__=="__main__":
    main()