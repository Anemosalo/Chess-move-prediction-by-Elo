# Learning stronger players from few games: pretraining on weak players

*Alexandros Anemos Aloupis · October 2026 · follows write-up 1 (the base model)*

## The question

In chess, games between very high-rated players are much rarer than games at average ratings, so there is relatively little data to train on. Can we make up for the lack of data by pretraining on average-rated games first and then fine-tuning on stronger players? How many strong-player games does pretraining on weaker players save, and what does the adaptation to stronger players cost on the weak players the model was built for?

## Setup

- 50,000 Lichess blitz games with both players rated 2000–2199 (September 2026), split by game: 80% training, 10% validation, 10% test.
- Pretraining starting point: the base model from write-up 1 (459,145 weights), trained only on 1000–1199 rated games.
- N: the number of strong-player games the model was trained on, which goes from 100 → 1,000 → 10,000 and finally all 40,019. For each N, 3 options were trained on the same games:

| Option | Starts from | Weights trained | Learning rate |
|---|---|---|---|
| scratch | random weights | all 459,145 | 10⁻³ |
| full fine-tune | the base model | all 459,145 | 10⁻⁴ |
| BN + output layer | the base model | only the BatchNorm scale/shift (1,664) and the output layer (4,745): 6,409 | 10⁻³ |

- Training: each run keeps its best validation score as a checkpoint and stops when it stops improving on validation. The validation set is the same for every N: 42,433 positions from 540 strong-player games.
- Different seeds: N = 100 and 1,000 were repeated with 3 random seeds, which means different games and shuffling. That came out to 24 runs in total and 4.5 GPU-hours on an Apple M1.
- Test measurement: the test set was measured once, at the end of training, on 37,050 strong-player positions and 28,654 weak-player positions, in order to also measure how much the model forgot about weak players' moves.

## Test results: bits per move on 2000–2199 (± = std over 3 seeds)

| N strong games | scratch | full fine-tune | BN + output layer |
|---|---|---|---|
| 0 (base model unchanged) | | 3.290 | 3.290 |
| 100 | 4.323 ± 0.024 | **3.081 ± 0.006** | 3.144 ± 0.004 |
| 1,000 | 3.768 ± 0.024 | **2.971 ± 0.001** | 3.087 ± 0.002 |
| 10,000 | 3.052 | **2.834** | 3.074 |
| all 40,019 | 2.841 | **2.775** | 3.076 |

The uniform baseline is 4.694. See figures F1 (this table as curves) and F3 (forgetting).

## Findings

1. Pretraining on weak players is worth a lot of strong-player games, compared with training from scratch with the same settings. Of course this holds mostly when N is small, but even with large N the pretrained model did slightly better.
   - The base model scores as well on the 2000–2199 band as a model trained from scratch on ~4,700 strong-player games.
   - Fine-tuned on only N = 100 games, it matches training from scratch on ~9,100 games (about 90× fewer games), and fine-tuned on N = 1,000 games it matches scratch on ~17,000 (about 17×).
2. Full fine-tuning beat the from-scratch models at every N. The size of the advantage of course shrank as N got larger: a 1.24-bit difference at N = 100 and a 0.07-bit difference at N = 40,019 (all games).
3. The results did not depend much on which games were randomly selected: for the fine-tunes, the three seeds differed by at most 0.011 bits, with a std of at most 0.006.
4. Fine-tuning only the BatchNorm scale and shift and the output layer is not enough at small N.
   - This option covers about 1.4% of the weights, most of them in the output layer. It gained at most 0.22 bits, and at 100 and 1,000 games it leveled off at 3.09–3.14 bits with a flat training loss.
   - At 10,000 games or more its training loss rose, so the optimization did not converge at a learning rate of 10⁻³.
5. The model forgot little, and the forgetting did not increase with more strong-player data.
   - On weak players, full fine-tuning loses 0.14 bits for N = 100, 0.26 for N = 1,000, 0.16 for N = 10,000 and 0.09 for all games. The spread at 1,000 games was large (std ± 0.064), and 10,000 and all games were run with only one seed.
   - The models trained from scratch on the 2000–2199 band predicted weak players' moves wrongly with high confidence, resulting in 3.4 to 6.0 bits.

## Relation to learning with few labels

- The idea of training only a few parameters of a frozen pretrained network comes from meta-transfer learning (Sun, Liu, Chua & Schiele, CVPR 2019). That work meta-learns a scaling and shifting for each neuron and finds it better than full fine-tuning with 1–5 examples per class. My BN + output layer option was a test loosely inspired by it: it trained BatchNorm's scale and shift and the output layer by plain gradient descent, without meta-learning, and most of its weights are in the output layer. With hundreds to thousands of games, full fine-tuning won instead, which kind of makes sense considering how few weights the option changes and the limited testing I did.
- Measuring old and new together is related to generalized few-shot learning (Kukleva, Kuehne & Schiele, ICCV 2021). There, a model must learn new classes from a few examples without forgetting the old ones. In my case the classes (the 4,168 possible moves) stayed the same; what changed is which moves players choose, so it is a shift in the data distribution rather than a shift to new classes.

## Limitations

- Only one seed at N = 10,000 and 40,019, and no intervals for test-set noise.
- The largest runs stopped while still improving.
- The BN option also updates BatchNorm's running statistics.

All numbers come from `results/all_runs.csv` (tables in `results/tables.md`, figures F1–F7 in `results/figures/`).

AI use: the evaluation and analysis scripts and the figures were made with Claude, based on my instructions.
