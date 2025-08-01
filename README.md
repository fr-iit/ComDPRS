# Com-DPRS: A Model-Agnostic Framework for Recommendation Using Composite Differential Privacy

## Abstract

Privacy concerns in recommender systems have grown with the increasing sensitivity of user interaction data and the enforcement of stricter data protection regulations. While Differential Privacy (DP) provides a principled framework for safeguarding user data, existing DP-based perturbation techniques often suffer from unbounded and biased outputs due to noise injection, resulting in substantial utility loss. In this paper, we propose Com-DPRS, a model-agnostic input perturbation framework based on Composite Differential Privacy (Com-DP) that generates bounded and unbiased perturbations tailored to structured user–item interaction data. To further improve utility, we introduce a hybrid optimization strategy that jointly minimizes theoretical variance and empirical loss, achieving an effective privacy–utility trade-off. We evaluate Com-DPRS on four real-world datasets using three representative collaborative filtering models (UserKNN, ALS, and EINMF), demonstrating its model-agnostic applicability. Experimental results show that Com-DPRS consistently outperforms state-of-the-art DP-based perturbation methods in both prediction accuracy (MSE) and ranking quality (HR@10) across varying privacy budgets. Furthermore, distortion analysis confirms that Com-DPRS introduces significantly less accuracy loss compared to existing approaches. These findings establish Com-DPRS as a robust, scalable, and privacy-preserving solution for practical recommendation scenarios.

## Requirements
  * Python 3
  * Sklearn
  * Numpy
  * Pandas
  * Matplotlib
  * Torch

## Instruction

  * You must have the MovieLens (https://grouplens.org/datasets/movielens/), yahoo movie (https://webscope.sandbox.yahoo.com/) and Amazon All_Beauty data (https://amazon-reviews-2023.github.io/) downloaded in your project.
  * Use 1. 'ml100k' ; 2. 'ml1m' ; 3. 'yahoo' ; 4. 'beauty' keywords as the value of dataset_name variable to load respective datset through DataLoader.py
  * Use epsilion = {0.1, 1.0, 2.0, 3.0, 4.0, 5.0}
  * First, run RecSys_GetParam.py to obtain hyperparameter values, then execute RecSys_InputPerturbation_ComDP.py using those values.

