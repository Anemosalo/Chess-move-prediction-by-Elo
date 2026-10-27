# Testing on all rating bands, and what the rating input actually learned

*Alexandros Anemos Aloupis · October 2026 · follows write-ups 1 (the base model) and 2 (the few-games experiment)*

## The question

The models so far were trained on only one or two bands (1000–1199 and 2000–2199). This raises two questions: how do the models perform on bands they never saw, and does the rating input actually act as a skill dial or not?

## Setup

- 5,000 more Lichess blitz games were added to the data for each of the rating bands 1200–1399, 1400–1599, 1600–1799 and 1800–1999. Here they are used for evaluation, not for training. Every band is measured on the test games of its first shard: 28,654 to 37,050 positions per band.
- Models measured:
  - the base model, trained only on 1000–1199;
  - the fine-tuned model, tuned on all 40,019 games of 2000–2199;
  - the scratch model, trained on all games of the 2000–2199 band.
- Rating test: each band's validation positions were measured again by each model, with the rating plane set to one value for every position (1000, 1100, … 2200), to find the rating that predicts that band best for each model.

## Results

Test bits per move with the real ratings (lower is better):

| Model | 1000–1199 | 1200–1399 | 1400–1599 | 1600–1799 | 1800–1999 | 2000–2199 |
|---|---|---|---|---|---|---|
| base (1000–1199 only) | **2.831** | **2.881** | 2.921 | 2.993 | 3.102 | 3.290 |
| base + fine-tune on all 2000–2199 | 2.925 | 2.883 | **2.819** | **2.766** | **2.744** | **2.775** |
| scratch on 2000–2199 only | 3.379 | 3.177 | 3.000 | 2.904 | 2.831 | 2.841 |
| *uniform over legal moves* | *4.604* | *4.640* | *4.675* | *4.687* | *4.707* | *4.694* |

The fine-tunes on 1,000 and 10,000 games fall between these rows (see figure F8).

The rating that predicts each band best (figure F9):

| Model | 1000–1199 | 1200–1399 | 1400–1599 | 1600–1799 | 1800–1999 | 2000–2199 |
|---|---|---|---|---|---|---|
| base | 1100 | 1100 | 1200 | 1200 | 1200 | 1200 |
| base + fine-tune on all 2000–2199 | 2000 | 2000 | 2000 | 2000 | 2100 | 2100 |
| scratch on 2000–2199 | 2100 | 2100 | 2000 | 2000 | 2100 | 2100 |

## Conclusion

- The base model: its predictions degrade smoothly as players get stronger, with its top-1 accuracy falling from 41.3% to 33.4%.
- The fine-tuned model: it is the best of all the models tested on the 1400–1999 range, although no model trained on those bands. It beats both single-band models there and pretty much ties the base model on 1200–1399.
- The main finding, though, is that each model predicts best when given the ratings it was trained on, even when the real ratings are different. That means the rating plane acts as "my training data" vs "unfamiliar input". Maia reported the same for its per-level models: each predicts best near its own training rating.
- Consequences:
  - The base model's predictions for strong players get worse when it is told their real ratings. Told 1200, it scores 3.066 bits, compared with 3.270 bits when told their real ratings (about 2,090).
  - Most of the fine-tuned model's forgetting on weak players comes from the rating input. It scores 2.921 bits with the real ratings and 2.849 when told 2000.
  - Learning two bands one after the other does not teach the model the meaning of the rating plane input. That is probably because no batch ever contained both rating levels, so nothing required the network to tell them apart.

## Next step

Train one model on several bands at once, with every batch mixing positions from all of them, and then repeat this test. This was the idea behind Maia-2. If the model learns to use the rating plane input as a skill dial, then each band's best rating should match its actual ratings, and the bands it never saw should be predicted well by interpolation.
This is tested in write-up 4.

All numbers come from `results/bands_eval.csv`, made by `eval_bands.py` (figures F8 and F9 in `results/figures/`).

AI use: the `eval_bands.py` script and the figure script were made with Claude.
