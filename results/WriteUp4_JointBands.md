# One model trained on several bands at once

*Alexandros Anemos Aloupis · October 2026 · follows write-ups 1–3*

## The question

Write-up 3 showed that training the model on different bands in sequence, rather than at the same time, did not teach it the meaning of the rating plane. Instead, it treated other ratings as unfamiliar input, which degraded its results.
What happens, then, if we train the model on several bands at once, in the same batches, and then test it on bands it was never trained on? Will it predict a band best when it is given that band's real rating?

## Setup

- Training data: 4 bands, 1000–1199, 1400–1599, 1800–1999 and 2200–2399, with 30,000 Lichess blitz games each (September 2026): ~6.7M training positions.
- Evaluation-only data: 5,000 games each for the bands 1200–1399, 1600–1799 and 2000–2199, which are the gaps between the training bands, and 800–999 and 2400–2599, which lie outside them.
- The same rating input sweep as in write-up 3: every band's validation positions are scored again with the rating plane set to 800, 900, … 2600 for every position, to find the rating that predicts each band best.

## Results

| Band | Test bits per move (positions) | Top-1 | Rating that predicts it best | Ratings within 0.002 bits of the best | Real mean rating | Best earlier model (test bits) |
|---|---|---|---|---|---|---|
| 800–999 *(never seen)* | 2.883 (27,609) | 40.7% | 1000 | 900–1100 | 914 | |
| 1000–1199 | **2.751** (28,654) | 42.5% | 1200 | 1100–1300 | 1105 | 2.831 |
| 1200–1399 *(never seen)* | **2.754** (30,823) | 42.0% | 1400 | 1300–1500 | 1299 | 2.881 |
| 1400–1599 | **2.726** (32,411) | 42.6% | 1600 | 1500–1700 | 1503 | 2.819 |
| 1600–1799 *(never seen)* | **2.710** (34,054) | 42.7% | 1700 | 1600–1800 | 1695 | 2.766 |
| 1800–1999 | **2.712** (34,887) | 42.5% | 2000 | 1900–2100 | 1893 | 2.744 |
| 2000–2199 *(never seen)* | **2.750** (37,050) | 41.8% | 2200 | 2100–2300 | 2087 | 2.775 |
| 2200–2399 | 2.797 (37,760) | 40.8% | 2300 | 2200–2400 | 2289 | |
| 2400–2599 *(never seen)* | 2.799 (38,906) | 40.8% | 2400 | 2300–2500 | 2485 | |

The uniform baseline (every legal move equally likely) is 4.58–4.72 bits per move, so the model needs 37–42% fewer bits on every band. See figure F10.

## Conclusion

1. The best rating to assume now rises with the band's actual rating.
   - Each band's sweep curve is a clear U (0.09–0.22 bits from bottom to top), and the U moves with the band. This is also true for the three in-between bands the model never saw.
   - Telling the model the real rating is now as good as the best fixed rating, whereas before, the models performed best at one fixed rating for every band, and the real ratings cost up to 0.20 bits.
2. Never-seen bands score well and close to their trained neighbours on the test positions:
   - 1600–1799 (2.710) is as good as its trained neighbours or better (2.726 and 2.712);
   - 2000–2199 (2.750) is between its neighbours (2.712 and 2.797);
   - 1200–1399 (2.754) is slightly worse than both of its neighbours (2.751 and 2.726).
3. The joint-bands model has the best performance of all the models on every band they share, for example 2.750 on 2000–2199 against 2.775 for the model fine-tuned on 40,019 games of that band. The comparison is not entirely fair, though:
   - it trained for 13,108 steps, about 2× the base model's 6,729 (which was still improving), so on 1000–1199 it is partly a comparison of training length;
   - it trained on 60,000 games of the two bands next to 2000–2199;
   - some players of the never-seen bands also play in the training bands' games;
   - it is one seed, with no intervals for the differences.
4. Outside the training range the bits are the highest: 2.883 on 800–999 and 2.799 on 2400–2599.
5. The answer to write-up 3: learning rating levels together, in the same batches, makes the model learn what the rating input represents. Maia-2 found the same on a larger scale.

## Limitations

- One training run (one seed). The best checkpoint was at step 13,000 of 13,108; the validation score was still falling until then.
- The sweep's step is 100 rating points, and the bottom of each curve is flat at that resolution.
- Stronger bands also play faster time controls and longer games, and the rating input absorbs those differences too.
- The comparison with the earlier models is confounded by training length, neighbouring bands and shared players.
- The model saw more variety than the earlier ones: 4 levels, though fewer games per level.

Numbers come from `runs/joint_4bands/` and `results/bands_eval_joint_4bands.csv`.

AI use: the multi-band code in `train.py`, the evaluation, and the analysis and figure scripts were made with Claude.
