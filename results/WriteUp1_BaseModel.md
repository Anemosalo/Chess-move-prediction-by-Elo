# Predicting the moves of 1000–1199 rated blitz players with a small CNN

*Alexandros Anemos Aloupis · October 2026 · code and data pipeline: github.com/Anemosalo/Chess-move-prediction-by-Elo*

## The question

Can a small neural network predict well the move a 1000–1199 rated player will choose in a blitz game?
Two things were measured:

- Bits per move: −log₂(p), where p is the probability the model gave to the move that was played. This is (ideally) the number of bits a compressor using the model would need to store the move.
- Top-1 accuracy: the percentage of positions in which the model's top choice was the move actually played.

## Data

The data was gathered from the Lichess open database and consisted of the first 1.5 GB of the September 2026 games (4,596,934 games).
Only blitz games that ended normally or on time, with both players inside our rating range of 1000–1199, were kept. That came out to 141,333 games and 8,568,827 positions in total.

The data was split by game: 80% training, 10% validation and 10% test, so no game had positions in more than one split.

## Model and training

- The input: the input layer receives the board from the view of the player to move (the board is mirrored for Black) as an [18, 8, 8] tensor. The 18 planes hold the pieces, the castling rights, en passant and the rating of the player to move, scaled to (Elo − 1500)/400 so it comes out as a small value (−1.25 to −0.75 for this band).
- The network: a residual CNN. It has a 3×3 convolution stem, then 6 residual blocks of two 3×3 convolutions that take in and output 64 channels and pass through BatchNorm, and then a 1×1 convolutional head. The total comes out to 459,145 weights.
- Output: the model outputs one score for every possible from-square/to-square combination plus 72 underpromotions, which comes out to 4,168 possible moves in chess. Illegal moves are removed by masking them out before the softmax, and the loss is calculated as cross-entropy in bits.
- Training: Adam with a learning rate of 10⁻³, batches of 1,024 positions and one pass over the 6.87 million training positions. That came out to 6,729 steps, which took 28 minutes on an Apple M1. The checkpoint with the best validation score was kept.

## Results

(The test set was measured only once.)

| Test positions | Model | Uniform over legal moves | Top-1 | Random legal move |
|---|---|---|---|---|
| 1000–1199 (849,649) | **2.830 bits/move** | 4.604 bits | **41.5%** | 6.8% |
| 2000–2199, never seen in training (371,077) | 3.263 bits/move | 4.691 bits | 33.7% | 5.9% |

- The model performed exceptionally well for its size on its training band of 1000–1199, with 41.5% top-1 accuracy and 38.5% fewer bits per move than choosing uniformly. It is worth noting that the model was still improving when it stopped, with the last 3 validations at 2.841 → 2.824 → 2.820 (figure F6).
- Tested on strong players (the 2000–2199 band), the model did well too, with 33.7% top-1 accuracy. Its performance dropped a lot compared with its training band, but that is expected since it never saw any other band.

## Project context and possible limitations

- Prior work: Maia (McIlroy-Young et al., KDD 2020) and Maia-2 (NeurIPS 2024) predict human moves by rating level with bigger models trained on far more games. They report over 50% top-1 accuracy. Their numbers were measured a bit differently: they skip the first 10 plies, since they can be repetitive, and moves made with under 30 seconds on the clock. This keeps the predictions meaningful, instead of taking advantage of positions that are easier to predict (the start of the game) or more unpredictable and less meaningful to guess (low time). This project is basically a small-scale model similar to Maia, built and trained from scratch on a laptop, so it is not really a competitor but rather a personal project for learning.
- Limitations:
  - one training run;
  - one rating band, so the rating input was basically redundant and the model cannot use it yet;
  - training was stopped while the model was still slowly improving.
- Next steps: fine-tuning on strong players (write-up 2) and training on several bands at once (write-up 4).

Tools: Python, python-chess, NumPy, PyTorch.

AI use: I wrote most of the move encoding, the network and the training loop. Claude Code helped with saving the shards correctly and with the analysis and figure scripts, based on my instructions and my descriptions of what I wanted from the data analysis.
