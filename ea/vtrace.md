# IMPALA - A Practical Primer

## Table of Contents
1. [Summary](#summary)
2. [Motivation](#motivation)
3. [Prerequisites](#prerequisites)
4. [Overview of Reinforcement Learning Algorithms](#overview-of-reinforcement-learning-algorithms) 
5. [IMPALA Overview](#impala-overview)
6. [VTrace Intuition](#vtrace-intuition)
7. [VTrace Algorithm](#vtrace-algorithm)
8. [Implementation Details](#implementation-details)
9. [Appendix A: Citations](#appendix-a-citations)

## Summary
### Goals
The goals of this document are:

1. To provide a high-level overview of IMPALA, and how it compares to other RL algorithms.
2. Explain the V-Trace algorithm, which is the key innovation of IMPALA
3. Provide a minimal implementation of IMPALA in PyTorch that others can use as a starting point for their own projects.

### Target Audience
The target audience for this document are programmers who are already somewhat familiar with RL. Many fundamental concepts in RL are not explained here, but we do provide links to resources that explain them in more detail.

## Motivation
[IMPALA](https://arxiv.org/abs/1802.01561)
(
    **IM**portance
    weighted
    **A**ctor
    **L**earner
    **A**rchitectures
)
is an RL training algorithm that was released in 2018.
It's a fairly popular [model-free](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) RL algorithm, [by number of citations](#appendix-a).
The reason why IMPALA is popular is that it is:

1. Relatively simple to implement
2. Efficient to train in a distributed setting

IMPALA is an [off-policy algorithm](https://towardsdatascience.com/on-policy-v-s-off-policy-learning-75089916bc2f).
Off-policy algorithms enable greater parallelization, as they do not need to wait for the policy to be updated before collecting new data.
Additionally, using an off-policy algorithm means that old experiences can be reused, increasing sample efficiency.

However, off-policy algorithms are more difficult to train than on-policy algorithms.
We cannot naively use a standard on-policy gradient algorithm like [REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) or [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html), as the difference between the policy used to collect the data and the current policy causes training instabilities.
IMPALA solves this problem by using the V-Trace algorithm, which corrects for the difference between the two policies.

In 2023, IMPALA is not a state-of-the-art RL algorithm.
However, although there are other RL algorithms like [DreamerV3](https://arxiv.org/pdf/2301.04104v1.pdf) and [R2D2](https://www.deepmind.com/publications/recurrent-experience-replay-in-distributed-reinforcement-learning) that achieve higher scores on several benchmarks and have better sample efficiency, they are not as easy to implement as IMPALA.

Despite it not being SOTA, it represents a large improvement over [A3C](https://arxiv.org/abs/1602.01783), and is less difficult to implement. Many of the ideas in IMPALA (especially Importance Sampling) are used in other RL algorithms.

## Overview of Reinforcement Learning Algorithms
The main contribution that IMPALA brings to the table is the ability to train efficiently in a distributed setting. 

## Impala Overview

## VTrace Intuition

## Vtrace Math

## Appendix A

Number of citations for selected RL algorithms:
| Algorithm | Year Published | Citations as of 2023-03-26 |
| --- | --- | --- |
| [PPO](https://arxiv.org/abs/1707.06347) | 2017 | 10913 |
| [A3C](https://arxiv.org/abs/1602.01783) | 2016 | 8467 |
| [SAC](https://arxiv.org/abs/1801.01290) | 2018 | 4969 |
| [RAINBOW](https://arxiv.org/abs/1710.02298) | 2018 | 1921 |
| [IMPALA](https://arxiv.org/abs/1802.01561) | 2018 | 1199 |
| [Ape-X](https://arxiv.org/abs/1803.00933) | 2018 | 662 |
| [DreamerV2](https://arxiv.org/abs/2010.02193) | 2020 | 391 |
| [R2D2](https://www.deepmind.com/publications/recurrent-experience-replay-in-distributed-reinforcement-learning) | 2018 | 380 |
| [DreamerV3](https://arxiv.org/abs/2301.04104) | 2023 | 12 |
