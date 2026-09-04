# latent-factor-cf

A movie recommender system built to understand **collaborative filtering** from first principles — starting from the raw sparsity problem, through baseline models, to a tuned SVD-based matrix factorization model, all benchmarked against each other on real data.

## Motivation

Most recommender system tutorials jump straight to calling a library function. This project was built the opposite way: understand *why* each technique exists before using it, and always measure against a baseline rather than trusting a single model's output on faith.

## Dataset

[MovieLens Latest Small](https://grouplens.org/datasets/movielens/) — 100,000 ratings, ~9,700 movies, 610 users. Chosen deliberately over the full 25M-rating dataset for fast iteration during development; the pipeline is designed to scale to the larger dataset as future work.

The user-item matrix is **~98% sparse** — of all possible (user, movie) rating pairs, only a small fraction actually exist. This sparsity is the central problem every technique below addresses in a different way.

## Approach

### 1. Baselines
Before trusting any learned model, two naive baselines were implemented from scratch to establish a performance floor:
- **Global mean baseline** — predicts the average rating for every pair, ignoring user and movie identity entirely.
- **Bias baseline** — predicts `global mean + user bias + movie bias`, capturing individual rating tendencies (harsh/generous raters, broadly loved/disliked movies) without modeling any user-movie interaction.

### 2. Matrix Factorization (SVD)
Implemented using `scikit-surprise`'s `SVD` (a regularized, gradient-trained latent factor model in the style of the Netflix Prize-winning approaches). Each user and movie is represented as a learned vector in a shared latent space; predicted rating = dot product of the two vectors.

Hyperparameters (`n_factors`, `n_epochs`, `lr_all`, `reg_all`) were tuned via grid search with 3-fold cross-validation.

## Results

| Model                          | RMSE   |
|---------------------------------|--------|
| Global mean baseline             | 1.0488 |
| Bias baseline (user + item bias) | 0.9174 |
| Tuned SVD (matrix factorization) | 0.8613 |

**Key finding:** the majority of predictable signal in this dataset comes from individual rating tendencies (bias), not personalized interaction patterns — going from "know nothing" to "know rating bias" closes most of the gap. Full latent factor modeling adds a smaller but real improvement on top. The tuned SVD result (0.8613) sits within the range typically reported for SVD-family models on MovieLens 100k (~0.87–0.92), indicating the implementation is sound and reasonably optimized rather than under- or over-fit.

## What this project deliberately does *not* cover (yet)

- **User cold-start**: recommending for a brand-new user with no rating history requires a separate content-based approach (e.g. genre-vector matching, or bootstrapping a pseudo-user vector from the latent representations of a few seed movies) since the trained model has no way to place an unseen user in latent space.
- **Item cold-start**: the model can only ever recommend movies present in its training catalog. Handling out-of-catalog movies would require external metadata (e.g. TMDb) and is out of scope here.

Both are natural, well-understood next steps and are noted here explicitly as a scoping decision rather than an oversight.

## Stack

Python, pandas, numpy, scikit-surprise, scikit-learn

## Structure

```
Data/           MovieLens CSVs
notebooks/      Baseline.ipynb, Matrix factorization.ipynb
model/          saved trained model artifact
```
