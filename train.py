import torch
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import time
import json
import glob
import os
import argparse
#import matplotlib.pyplot as plt

def load_shard(path):
    d = np.load(path)
    return tuple(d[k] for k in ("planes","y","legal_flat","offsets","rating","split","game_id"))


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

def evaluate(model,rows,planes,y,legal_flat,offsets,rating,device):
    model.eval()
    total_bits= 0.0
    total_n = 0 #total positions
    correct = 0
    with torch.no_grad():
        for start in range(0,len(rows),1024):
            idx = rows[start:start+1024] #next chunk of row nums
            X, mask , target = get_batch(idx,planes,y,legal_flat,offsets,rating)
            X,mask,target = X.to(device),mask.to(device),target.to(device)
            logits = model(X)
            model_prediction = (logits+mask).argmax(dim=1)
            correct += (model_prediction==target).sum().item()
            loss = F.cross_entropy(logits+mask,target)/math.log(2)
            total_bits+=loss.item()*len(idx)
            total_n+=len(idx)
        model.train()
        return total_bits/total_n , correct/total_n


def parse_args():
    p = argparse.ArgumentParser(description="Training or evaluation of the move prediction network")
    p.add_argument("--data",nargs="+",default=["1000-1199"],help="the rating band(s) to train on for example 1000-1199, or several: 1000-1199 1400-1599 (then every batch mixes all of them)")
    p.add_argument("--init",default="scratch",help="'scratch' or a path to the saved weights from previous training")
    p.add_argument("--tune",default="all", choices=["all","bn"],help="which weights are allowed to change. 'all' by default and 'bn' for only the Batch Norms γ and β")
    p.add_argument("--games",type=int,default=None,help="Use only the first N training games")
    p.add_argument("--epochs",type=int,default=1,help="Number of epochs on the training data")
    p.add_argument("--lr",type=float,default=1e-3,help="Learning rate, 1e-3 by default")
    p.add_argument("--seed",type=int,default=0,help="Seed to get identical results on same seeds")
    p.add_argument("--eval-every",type=int,default=200,help="After how many training steps to evaluate. 200 by default")
    p.add_argument("--also",nargs="*",default=[],help="What extra bands to also mesure on, for example 1000-1199")
    p.add_argument("--out",required=True,help="run folder name inside of runs/")
    p.add_argument("--test",action="store_true",help="mesure of the best weights once on the test split set at the end")
    p.add_argument("--eval-only",action="store_true",help="No training just mesure --init ont --data and --also")
    p.add_argument("--patience", type=int,default=None,help="Stop after this number of evaluations if val is not improving. default = None means never stop early")
    p.add_argument("--test-shards", type=int, default=1, help="use the test rows of the first N shards (99 = all)")
    return p.parse_args()


def band_paths(band):
    return sorted(glob.glob(f"data/games_encoded/lichess_2026-09_elo{band}_blitz_*.npz"))

def take_rows(data,rows): ## smaller copy of a shard with only these rows (so a val set doesnt keep the whole ~380MB shard in memory)
    planes,y,legal_flat,offsets,rating,split,game_id = data
    lengths = offsets[rows+1]-offsets[rows] # number of legal moves of each row
    new_offsets = np.concatenate([[0],np.cumsum(lengths)]).astype(np.int64)
    new_flat = np.concatenate([legal_flat[offsets[r]:offsets[r+1]] for r in rows]).astype(np.int16)
    return (planes[rows],y[rows],new_flat,new_offsets,rating[rows],split[rows],game_id[rows])

def load_val(band):
    data = load_shard(band_paths(band)[0])
    rows = np.flatnonzero(data[5]==1)
    small = take_rows(data,rows) # same rows in the same order, so the val numbers dont change
    return np.arange(len(rows)),small

def train_plan(band,n_games): ## we do all this suffling instead of getting the first n in order for the seed to have impact and give random results for different seeds
    plan = []
    remaining = n_games
    for path in band_paths(band):
        d = np.load(path)
        split,game_id = d["split"],d["game_id"]
        train_rows = np.flatnonzero(split==0)
        if remaining is not None:
            games_here = np.unique(game_id[train_rows]) ## Its the training games in this shard
            keep = np.random.permutation(games_here)[:remaining] # random selections of them with at most remaining
            train_rows = train_rows[np.isin(game_id[train_rows],keep)]
            remaining -=len(keep)
        plan.append((path,train_rows))
        if remaining==0: ## got all games we want so break and return
            break

    return plan

#-------------#
# Several bands: every batch takes the same number of positions from each band
# One "stream" per band walks over that band's training shards in a random order, a chunk of rows at a time.
# Only one shard per band is in memory at a time.

def new_stream(band,n_games):
    plan = train_plan(band,n_games)
    return {"band":band,"plan":plan,"queue":[],"data":None,"order":None,"pos":0,"passes":0}

def next_rows(stream,n): ## the next n training rows of this band (fewer at the end of a shard) and the shard they come from
    while stream["order"] is None or stream["pos"]>=len(stream["order"]):
        if len(stream["queue"])==0: # every shard of this band used: start a new pass over the band in a new random order
            stream["queue"] = list(np.random.permutation(len(stream["plan"])))
            stream["passes"]+=1
        path,rows = stream["plan"][stream["queue"].pop()]
        stream["data"] = load_shard(path)
        stream["order"] = np.random.permutation(rows)
        stream["pos"] = 0
    idx = stream["order"][stream["pos"]:stream["pos"]+n]
    stream["pos"]+=n
    return idx,stream["data"]

def get_mixed_batch(streams,B): ## a batch of about B positions, B/len(streams) from every band
    per_band = B//len(streams)
    Xs,masks,ys = [],[],[]
    for s in streams:
        idx,data = next_rows(s,per_band)
        X,mask,y = get_batch(idx,*data[:5])
        Xs.append(X)
        masks.append(mask)
        ys.append(y)
    return torch.cat(Xs),torch.cat(masks),torch.cat(ys) # (B,18,8,8) , (B,4168) , (B)
#-------------#

def evaluate_split(model, band, split_value, n_shards, device):
    total_bits, total_top1, total_n = 0.0, 0.0, 0
    for path in band_paths(band)[:n_shards]:
        data = load_shard(path)
        rows = np.flatnonzero(data[5] == split_value)
        bits, top1 = evaluate(model, rows, *data[:5], device)
        total_bits += bits * len(rows)          # evaluate returns means, so × count = sums
        total_top1 += top1 * len(rows)
        total_n += len(rows)
    return total_bits / total_n, total_top1 / total_n, total_n

def run_test(model, bands, n_shards, run_dir, device):
    """Measures the model on the test split of each band and saves the results to run_dir/test.json."""
    test = {}
    for band in bands:
        bits, top1, n = evaluate_split(model, band, 2, n_shards, device)
        test[band] = {"bits": bits, "top1": top1, "positions": n}
        print(f"TEST {band}: {bits:.3f} bits, top-1 {top1:.1%} ({n:,} positions)", flush=True)
    with open(os.path.join(run_dir, "test.json"), "w") as f:
        json.dump(test, f, indent=2)




def main():

    args = parse_args()

    run_dir = os.path.join("runs",args.out)
    os.makedirs(run_dir,exist_ok=False)

    with open(os.path.join(run_dir,"config.json"),"w") as f:
        json.dump(vars(args),f,indent=2)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bands = args.data # list of training bands, 1 band = the normal training like before
    if len(bands)==1:
        plan = train_plan(bands[0],args.games)
        n_positions = sum(len(rows) for _,rows in plan)
    else:
        streams = [new_stream(band,args.games) for band in bands] # --games = games per band
        n_positions = sum(len(rows) for s in streams for _,rows in s["plan"])
        print("training positions per band:", {s["band"]: sum(len(rows) for _,rows in s["plan"]) for s in streams}, flush=True)


    device = "cuda"if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu") 
    model = Network().to(device)
    if args.init !="scratch":
        model.load_state_dict(torch.load(args.init,map_location=device))


    if args.tune =="bn":
        if args.init=="scratch":
            raise SystemExit("--tune bn needs --init with already saved weights")
        for name, p in model.named_parameters():
            p.requires_grad = ("bn" in name) or name.startswith("exit_head")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable,lr=args.lr)
    print("trainable weights:", sum(p.numel() for p in trainable))

    all_bands = bands + [b for b in args.also if b not in bands] # training bands first, then the ones only measured
    vals = {band: load_val(band) for band in all_bands}

    if args.eval_only: # no training: measure --init on val (and test) and stop
        if args.init == "scratch":
            raise SystemExit("--eval-only needs --init with saved weights")
        for band, (rows, data) in vals.items():
            bits, top1 = evaluate(model, rows, *data[:5], device)
            print(f"VAL {band}: {bits:.3f} bits, top-1 {top1:.1%}", flush=True)
        if args.test:
            run_test(model, all_bands, args.test_shards, run_dir, device)
        return
    history = {"step":[], "train_bits":[], "epoch_end":[],"val":{band:{"step":[],"bits":[],"top1":[]}for band in vals},
               "val_mean":{"step":[],"bits":[]}} # val_mean = mean val bits of the training bands, picks the best checkpoint

    step = 0
    best = float("inf")
    bad_evals = 0
    stop = False
    window_bits,window_n =0.0,0
    B=1024


    


    def mesure():
        nonlocal best,bad_evals,stop 
        for band ,(rows,data) in vals.items():
            bits,top1 = evaluate(model,rows,*data[:5],device)
            h = history["val"][band]
            h["step"].append(step)
            h["bits"].append(bits)
            h["top1"].append(top1)
            print(f"step {step} | {band}: {bits:.3f} bits, top-1 {top1:.1%}", flush=True)

        current = sum(history["val"][band]["bits"][-1] for band in bands)/len(bands) # mean of the training bands last val (only the training bands, never the held out ones)
        history["val_mean"]["step"].append(step)
        history["val_mean"]["bits"].append(current)
        if len(bands)>1:
            print(f"step {step} | mean of training bands: {current:.3f} bits", flush=True)
        if current<best:
            best = current
            bad_evals = 0
            torch.save(model.state_dict(),os.path.join(run_dir,"best.pt"))
        else:
            bad_evals+=1
            if args.patience is not None and bad_evals>=args.patience:
                stop = True
        with open(os.path.join(run_dir,"history.json"),"w") as f:
            json.dump(history,f)


    def train_step(X,mask,target): ## one training step on one batch, the same for one band or several
        nonlocal step,window_bits,window_n
        X,mask,target = X.to(device) , mask.to(device), target.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits+mask,target)/math.log(2) #calculated with natural log so we divide by log(2) to get bits meaning log_2 calculation instead of log_e
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step+=1
        window_bits+=loss.item()
        window_n+=1
        if step%50==0:
            history["step"].append(step)
            history["train_bits"].append(window_bits/window_n)
            window_bits,window_n=0.0,0
        if step % args.eval_every == 0:
            mesure()


    mesure()
    if len(bands)==1: ## one band: like before, shard by shard
        cache = {} if len(plan)<=3 else None
        for epoch in range(args.epochs):
            t = time.time()
            for i in np.random.permutation(len(plan)):
                path,rows = plan[i]
                if cache is not None and path in cache:
                    data = cache[path]
                else:
                    data = load_shard(path)
                    if cache is not None:
                        cache[path] = data
                planes,y,legal_flat,offsets,rating = data[:5]
                order = np.random.permutation(rows)
                for start in range(0,len(order),B):
                    idx = order[start:start+B]
                    X, mask, target = get_batch(idx, planes, y, legal_flat, offsets, rating)
                    train_step(X,mask,target)
                    if stop:
                        break
                if stop:
                    break
            history["epoch_end"].append(step)
            print(f"epoch {epoch} done ({time.time()-t:.0f})s",flush=True)
            if stop:
                print("Early stopping: eval stoped improving")
                break
    else: ## several bands: every batch mixes all of them
        steps_per_epoch = n_positions//B # one epoch = as many steps as one pass over all the training positions
        print(f"{len(bands)} bands, {n_positions:,} training positions, {steps_per_epoch:,} steps per epoch", flush=True)
        for epoch in range(args.epochs):
            t = time.time()
            for _ in range(steps_per_epoch):
                X, mask, target = get_mixed_batch(streams,B)
                train_step(X,mask,target)
                if stop:
                    break
            history["epoch_end"].append(step)
            print(f"epoch {epoch} done ({time.time()-t:.0f})s | passes started per band:", {s["band"]: s["passes"] for s in streams}, flush=True)
            if stop:
                print("Early stopping: eval stoped improving")
                break
    if step% args.eval_every !=0:
        mesure() #mesure the final state too
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f)

    if args.test: # the best weights (not the last ones), measured once on the test split
        model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
        run_test(model, all_bands, args.test_shards, run_dir, device)




if __name__=="__main__":
    main()