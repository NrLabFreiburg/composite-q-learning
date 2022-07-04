# Robust and Data-efficient Q-learning by Composite Value-estimation

This repository is the official implementation of [Robust and Data-efficient Q-learning by Composite Value-estimation](https://openreview.net/forum?id=ak6Bds2DcI). 


Abstract: In the past few years, off-policy reinforcement learning methods have shown promising results in their application to robot control. Q-learning based methods, however, still suffer from poor data-efficiency and are susceptible to stochasticity or noise in the immediate reward, which is limiting with regard to real-world applications. We alleviate this problem by proposing two novel off-policy Temporal-Difference formulations: (1) Truncated Q-functions which represent the return for the first n steps of a target-policy rollout with respect to the full action-value and (2) Shifted Q-functions, acting as the farsighted return after this truncated rollout. This decomposition allows us to optimize both parts with their individual learning rates, achieving significant learning speedup and robustness to variance in the reward signal, leading to the Composite Q-learning algorithm. We show the efficacy of Composite Q-learning in the
tabular case and furthermore employ Composite Q-learning within TD3. We compare Composite TD3 with TD3 and TD3(Delta), which we introduce as an off-policy variant of TD(Delta). Moreover, we show that Composite TD3 outperforms TD3 as well as TD3(Delta) significantly in terms of data-efficiency in multiple simulated robot tasks and that Composite Q-learning is robust to stochastic immediate rewards.

## Cite

```cite
@article{
	kalweit2022robust,
	title={Robust and Data-efficient Q-learning by Composite Value-estimation},
	author={Gabriel Kalweit and Maria Kalweit and Joschka Boedecker},
	journal={Transactions on Machine Learning Research},
	year={2022},
	url={https://openreview.net/forum?id=ak6Bds2DcI}
}
```

## Contributing

>📋  Awesome that you are interested in our work! Please write an e-mail to kalweitg@cs.uni-freiburg.de
