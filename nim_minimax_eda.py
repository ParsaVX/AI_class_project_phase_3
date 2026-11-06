"""Nim minimax / alpha-beta and reproducible EDA.
Run: python nim_minimax_eda.py [--games 200] [--seed 0] [--outdir .]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

START = (3, 4, 5)
nodes = 0

def moves(piles, order="small_first"):
    """Return every board reachable by removing >=1 stones from one pile."""
    result = []
    for i, n in enumerate(piles):
        takes = range(1, n + 1) if order == "small_first" else range(n, 0, -1)
        for take in takes:
            new = list(piles)
            new[i] -= take
            result.append(tuple(new))
    return result

def minimax(piles, player):
    """Perfect-play utility from A's perspective: +1 A, -1 B."""
    global nodes
    nodes += 1
    if sum(piles) == 0:
        return -1 if player == "A" else 1
    vals = [minimax(nxt, "B" if player == "A" else "A") for nxt in moves(piles)]
    return max(vals) if player == "A" else min(vals)

def alphabeta(piles, player, alpha=-2, beta=2, order="small_first"):
    """Minimax with alpha-beta pruning; nodes counts visited search positions."""
    global nodes
    nodes += 1
    if sum(piles) == 0:
        return -1 if player == "A" else 1
    if player == "A":
        value = -2
        for nxt in moves(piles, order):
            value = max(value, alphabeta(nxt, "B", alpha, beta, order))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = 2
    for nxt in moves(piles, order):
        value = min(value, alphabeta(nxt, "A", alpha, beta, order))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value

def best_move(piles, player):
    candidates = moves(piles, "big_first")
    if not candidates:
        return None
    best, best_value = None, None
    for nxt in candidates:
        val = alphabeta(nxt, "B" if player == "A" else "A", -2, 2, "big_first")
        if best_value is None or (val > best_value if player == "A" else val < best_value):
            best, best_value = nxt, val
    return best

def play_game(piles, rng):
    board, player, count = piles, "A", 0
    while sum(board):
        if player == "A":
            board = best_move(board, "A")
        else:
            opts = moves(board)
            board = opts[int(rng.integers(len(opts)))]
        player = "B" if player == "A" else "A"
        count += 1
    return ("B" if player == "A" else "A"), count

def build_data(n_games=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_games):
        piles = tuple(int(x) for x in rng.integers(1, 5, size=3))
        global nodes
        nodes = 0
        start_value = alphabeta(piles, "A", -2, 2, "big_first")
        n_ab = nodes
        nodes = 0
        mm_value = minimax(piles, "A")
        n_mm = nodes
        assert mm_value == start_value, f"Search mismatch for {piles}"
        winner, n_moves = play_game(piles, rng)
        rows.append({"p1": piles[0], "p2": piles[1], "p3": piles[2],
            "total": sum(piles), "nim_sum": piles[0] ^ piles[1] ^ piles[2],
            "first_wins": start_value > 0, "nodes_minimax": n_mm,
            "nodes_alphabeta": n_ab, "moves": n_moves, "winner": winner})
    return pd.DataFrame(rows)

def make_plots(df, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    df["nodes_minimax"].plot(kind="hist", bins=20, ax=axes[0,0], title="Histogram: minimax nodes")
    df.groupby("total")["first_wins"].mean().plot(kind="bar", ax=axes[0,1], title="A win rate by total stones")
    df[["nodes_minimax", "nodes_alphabeta"]].plot(kind="box", logy=True, ax=axes[1,0], title="Box plot (log scale)")
    axes[1,1].scatter(df["total"], df["nodes_minimax"], s=8)
    axes[1,1].set_yscale("log"); axes[1,1].set_title("Total stones vs nodes")
    plt.tight_layout(); fig.savefig(outdir / "nim_eda.png", dpi=150); plt.close(fig)

def make_animation(outdir, seed=1):
    rng = np.random.default_rng(seed)
    board, player = START, "A"
    history = [(board, "start")]
    while sum(board):
        old = board
        board = best_move(board, "A") if player == "A" else moves(board)[int(rng.integers(len(moves(board))))]
        i = next(k for k in range(3) if board[k] != old[k])
        history.append((board, f"{player} takes {old[i]-board[i]} from pile {i}"))
        player = "B" if player == "A" else "A"
    winner = "B" if player == "A" else "A"
    fig, ax = plt.subplots(figsize=(4,3), dpi=70)
    def draw(t):
        piles, text = history[min(t, len(history)-1)]
        ax.clear(); ax.bar(["pile 0","pile 1","pile 2"], piles, color=["#4f46e5","#ec4899","#f59e0b"])
        ax.set_ylim(0, max(START)+1)
        ax.set_title(text + ("   winner: "+winner if t >= len(history)-1 else ""), fontsize=9)
    anim = FuncAnimation(fig, draw, frames=len(history)+3, interval=800)
    path = outdir / "nim_game.gif"
    anim.save(path, writer=PillowWriter(fps=1.25)); plt.close(fig)
    return winner, len(history)-1, path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)
    global nodes
    nodes=0; v1=minimax(START,"A"); n_mm=nodes
    nodes=0; v2=alphabeta(START,"A",-2,2,"small_first"); n_ab=nodes
    nodes=0; v3=alphabeta(START,"A",-2,2,"big_first"); n_good=nodes
    assert v1 == v2 == v3
    print(f"START values: {v1}, {v2}, {v3}; nodes minimax/natural/big-first: {n_mm}/{n_ab}/{n_good}")
    assert len(moves((1,2))) == 3 and (0,2) in moves((1,2))
    df=build_data(args.games,args.seed)
    assert list(df.columns)==["p1","p2","p3","total","nim_sum","first_wins","nodes_minimax","nodes_alphabeta","moves","winner"]
    assert len(df)==args.games and not df.isna().any().any()
    assert ((df.nim_sum != 0) == df.first_wins).all()
    csv=args.outdir/"nim_games.csv"; df.to_csv(csv,index=False)
    # Reload the saved artifact to verify file contents, shape, and schema.
    check=pd.read_csv(csv)
    assert check.equals(df), "CSV did not round-trip exactly"
    make_plots(df,args.outdir)
    winner,nmoves,gif=make_animation(args.outdir)
    print(f"CSV verified: {csv} ({len(check)} rows x {len(check.columns)} columns)")
    print(f"Missing={int(check.isna().sum().sum())}; duplicated={int(check.duplicated().sum())}")
    print(f"A first-player win rate={df.first_wins.mean():.1%}; smart-A game win rate={(df.winner=='A').mean():.1%}")
    print(f"nim_sum theorem agreement={((df.nim_sum!=0)==df.first_wins).mean():.1%}")
    print(f"Alpha-beta node ratio={df.nodes_alphabeta.sum()/df.nodes_minimax.sum():.2%}")
    print(f"Animation verified: {gif}; {nmoves} moves; winner {winner}")
    print(f"Plot: {args.outdir/'nim_eda.png'}")

if __name__ == "__main__":
    main()
