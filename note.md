## Perceptual Aliasing

In the huggingface course [advantages and disadvantages PG method](https://huggingface.co/learn/deep-rl-course/unit4/advantages-disadvantages#policy-gradient-methods-can-learn-a-stochastic-policy), It refers to the problem of perceptual aliasing:

>  Perceptual aliasing is when two states seem (or are) the same but need different actions.

However, I think the emergence of perceptual aliasing is just because the game setting is not a Markov process. In a Markov process, identical states makes no difference to each other, since the future is exactly related with current state and has no relationship with the past actions or states.