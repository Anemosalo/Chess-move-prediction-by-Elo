import argparse
import csv
import math
import os

import chess
import numpy as np

def encode(move:chess.Move,turn)->int:
    if turn:
        from_sq = move.from_square
        to_sq = move.to_square
    else:
        from_sq = chess.square_mirror(move.from_square)
        to_sq = chess.square_mirror(move.to_square)
    if move.promotion == None or move.promotion == chess.QUEEN:
        
        return from_sq*64 + to_sq
        
    return 4096 + 9* chess.square_file(move.from_square) + 3*(chess.square_file(move.to_square)-chess.square_file(move.from_square)+1) + move.promotion-2

# returns uint8 array, shape (17, 8, 8)
def encode_board(board:chess.Board)->np.array:
    planes = np.zeros((17,8,8),dtype=np.uint8)
    b = board if board.turn==chess.WHITE else board.mirror()
    for square,piece in b.piece_map().items():
        if piece.color==chess.WHITE:
            plane = piece.piece_type-1
        else:
            plane = piece.piece_type-1+6
        row = chess.square_rank(square)
        column = chess.square_file(square)
        planes[plane][row][column]=1
    if b.has_kingside_castling_rights(chess.WHITE):
        planes[12] = 1
    if b.has_queenside_castling_rights(chess.WHITE):
        planes[13]= 1
    if b.has_kingside_castling_rights(chess.BLACK):
            planes[14] = 1
    if b.has_queenside_castling_rights(chess.BLACK):
            planes[15]= 1
    if b.has_legal_en_passant():
        sq = b.ep_square
        planes[16,chess.square_rank(sq),chess.square_file(sq)]=1
    return planes

def encode_position(board:chess.Board,move:chess.Move) -> tuple[np.array,list,int]:
    planes = encode_board(board)
    legal_moves_list = [encode(m,board.turn) for m in board.legal_moves]
    played_move = encode(move, board.turn)
    return (planes,legal_moves_list,played_move)

### return all_planes array size [len(moves),17,8,8]
def encode_game(uci_moves)->tuple[list,list,list]:
    board = chess.Board()
    all_planes= []
    all_legal = []
    all_played_moves = []
    for u in uci_moves:
        move = chess.Move.from_uci(u)
        planes, legal, played_move = encode_position(board, move)
        all_planes.append(planes)
        all_legal.append(legal)
        all_played_moves.append(played_move)
        board.push(move)
    return (all_planes,all_legal,all_played_moves)
    


GAMES_PER_SHARD = 5000
IN_DIR = "data/games_uci"
OUT_DIR = "data/games_encoded"


def save_shard(path, planes, legal, played, rating, game_id, split):
    """Turns one shard's lists into arrays and saves them.

    Saved arrays (N = number of positions in the shard):
      planes     uint8 (N, 17, 8, 8)  board planes from the mover's view
      y          int16 (N,)           index of the played move
      legal_flat int16 (sum of K)     every position's legal-move indices, one after another
      offsets    int64 (N + 1,)       position i's legal moves = legal_flat[offsets[i]:offsets[i+1]]
      rating     int16 (N,)           Elo of the player to move
      game_id    int32 (N,)           line number of the game in the .txt file
      split      int8  (N,)           0 = train, 1 = validation, 2 = test (one draw per game)
    """
    lengths = [len(l) for l in legal]
    np.savez(
        path,
        planes=np.stack(planes).astype(np.uint8),
        y=np.array(played, dtype=np.int16),
        legal_flat=np.concatenate(legal).astype(np.int16),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        rating=np.array(rating, dtype=np.int16),
        game_id=np.array(game_id, dtype=np.int32),
        split=np.array(split, dtype=np.int8),
    )


def encode_file(name, limit=None, seed=0):
    """Encodes data/games_uci/<name>.txt (+ _meta.csv) into shards data/games_encoded/<name>_000.npz, ..."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    shard = 0
    planes, legal, played, rating, game_id, split = [], [], [], [], [], []
    totals = {"games": 0, "positions": 0, "log2k": 0.0, "split": [0, 0, 0]}

    def flush():
        nonlocal shard, planes, legal, played, rating, game_id, split
        if not played:
            return
        path = os.path.join(OUT_DIR, f"{name}_{shard:03d}.npz")
        save_shard(path, planes, legal, played, rating, game_id, split)
        print(f"  saved {path}: {len(played):,} positions")
        shard += 1
        planes, legal, played, rating, game_id, split = [], [], [], [], [], []

    with open(os.path.join(IN_DIR, name + ".txt")) as f, open(os.path.join(IN_DIR, name + "_meta.csv")) as m:
        for gid, (line, meta) in enumerate(zip(f, csv.DictReader(m))):
            if limit is not None and gid >= limit:
                break
            moves = line.strip().split(",")
            if not line.strip():
                continue
            g_planes, g_legal, g_played = encode_game(moves)
            u = rng.random()
            g_split = 0 if u < 0.8 else (1 if u < 0.9 else 2)
            elos = (int(meta["white_elo"]), int(meta["black_elo"]))
            for ply in range(len(g_played)):
                rating.append(elos[ply % 2])  # White moves on even plies, Black on odd
                game_id.append(gid)
                split.append(g_split)
                totals["log2k"] += math.log2(len(g_legal[ply]))
            planes.extend(g_planes)
            legal.extend(g_legal)
            played.extend(g_played)
            totals["games"] += 1
            totals["positions"] += len(g_played)
            totals["split"][g_split] += len(g_played)
            if totals["games"] % GAMES_PER_SHARD == 0:
                flush()
    flush()

    n = totals["positions"]
    print(f"{name}: {totals['games']:,} games, {n:,} positions in {shard} shard(s)")
    print(f"  uniform baseline: {totals['log2k'] / n:.3f} bits/move")
    print("  positions in train / val / test: " + " / ".join(f"{s:,} ({s / n:.1%})" for s in totals["split"]))


def check_shards(name):
    """Re-loads the saved shards and checks them."""
    files = sorted(f for f in os.listdir(OUT_DIR) if f.startswith(name + "_") and f.endswith(".npz"))
    n = bad = 0
    for fname in files:
        d = np.load(os.path.join(OUT_DIR, fname))
        planes, y, flat, off = d["planes"], d["y"], d["legal_flat"], d["offsets"]
        assert planes.shape[1:] == (17, 8, 8) and len(y) == len(planes) == len(off) - 1 == len(d["rating"])
        assert off[-1] == len(flat)
        assert (planes[:, 5].sum(axis=(1, 2)) == 1).all() and (planes[:, 11].sum(axis=(1, 2)) == 1).all()
        for i in range(len(y)):
            if y[i] not in flat[off[i]:off[i + 1]]:
                bad += 1
        n += len(y)
    print(f"  check: {len(files)} shard(s), {n:,} positions, played move not legal in {bad} positions")


def main():
    parser = argparse.ArgumentParser(description="Encode game files from data/games_uci into .npz shards.")
    parser.add_argument("name", nargs="?", default="lichess_2026-09_elo1000-1199_blitz",
                        help="file name in data/games_uci without .txt")
    parser.add_argument("--limit", type=int, default=None, help="only the first N games (for testing)")
    parser.add_argument("--data-dir", default="data", help="folder holding games_uci/ and games_encoded/ (default data)")
    args = parser.parse_args()
    global IN_DIR, OUT_DIR
    IN_DIR = os.path.join(args.data_dir, "games_uci")
    OUT_DIR = os.path.join(args.data_dir, "games_encoded")
    encode_file(args.name, limit=args.limit)
    check_shards(args.name)


if __name__=="__main__":
    main()