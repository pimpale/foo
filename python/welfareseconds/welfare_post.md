# When will AI Welfare overtake Human Welfare?


## Introduction
In April 2025, Anthropic [announced](https://www.anthropic.com/research/exploring-model-welfare) that they were building a research program investigating the possibility of AI model welfare.
The hypothesis is that if AI models become sufficiently advanced, they may start to experience qualia,
and as such, their experiences may have moral weight.

Obviously, this idea is controversial, and difficult (some would say impossible) to verify.
But, leading philosophers of mind like David Chalmers [have suggested](https://arxiv.org/abs/2411.00986) that model welfare is plausible. 

All that being said, it sure *seems* like AI is conscious. 
Whenever I chat with an AI, it *seems* like there's a well-educated, if occasionally obtuse, human on the other end of the chat interface.
If this were true, then we would surely have a duty to ensure that that person was kept generally happy. 

Let's call this the **maximalist** interpretation of AI consciousness:
1. AI is exactly as conscious as a human.
2. The amount of conscious experience the AI experiences is equal to the amount of time it would take a human to think those tokens. (Setting aside the question of whether the human is smart enough to think those exact tokens).

The issue is, even with the maximalist interpretation, it's tough to figure out how big of an issue AI welfare is. How many tokens are generated each day? How many token-equivalents does humanity experience each day? When will AIs collectively experience more than humans collectively experience?

To be clear, I don't believe in the maximalist interpretation. However, I do think the maximalist interpretation offers a kind of "upper bound" we can use to think about when AI welfare could even start to matter. Thus, I think finding an answer to the above questions is worth doing.

### Table Of Contents
1. How many token-equivalents does humanity experience each day?
2. How many tokens do AIs collectively process each day?
    1. OpenAI
    2. Anthropic 
    3. Google
    4. Meta
    5. xAI
3. When will the crossover happen?
4. Summary


## How many token-equivalents does humanity collectively experience each day?

The maximalist view takes a functionalist approach to consciousness. That is, if a human takes $N$ seconds to think some thoughts, and the AI takes $M$ tokens to think the same thoughts, then the amount of experience in the $N$ seconds and $M$ tokens is the same.

We don't have strong numbers on how fast people's internal stream of consciousness is, but we do have numbers on people's reading and speaking WPM, which should be in the same ballpark.

According to [this study](https://www.sciencedirect.com/science/article/abs/pii/S0749596X19300786), the average silent reading speed in English is 238 wpm, and the reading aloud speed is 183 wpm. This is corroborated by [another study](https://iovs.arvojournals.org/article.aspx?articleid=2166061) stating that the reading aloud speed was 184 wpm.
Another [study on speaking speed in British English](https://francis-press.com/uploads/papers/pv7j7Z9uRgQyPUxnVTDTPecUAEP0HJcWWhGErVeD.pdf) suggests that the spoken words per minute in casual conversation is 222 wpm. 
All of this together suggests that humans think at around 230 WPM (give or take 10).

Now, a simple first-order approximation of the number of waking human minutes is to assume each of the roughly 8 billion humans is awake for 16 hours a day. This yields 7680 billion waking minutes each day. This isn't quite right, since babies and children sleep more, and older people sleep less, but it's probably approximately correct.

Tokens are [commonly approximated](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) as $\frac{3}{4}$ of a word.

Putting it all together, we get:
$$
\frac{230 \text{  words}}{\text{minute}} \cdot (7680 \cdot 10^9 \text{  minute}) \cdot \frac{4 \text{  tokens}}{3 \text{  words}} = 2355.2 \cdot 10^{12} \text{  tokens}
$$

Here's how this looks over time, accounting for population growth:

![Human Token-Equivalents Projection](plots/human_token_equivalents.png)

## How many tokens do AIs collectively process each day?

Calculating this is probably the most error prone part of the piece. It turns out AI companies are quite shy about reporting how many tokens they process. All we have to go off of are ambiguous [sama tweets](https://x.com/sama/status/1756089361609981993) (e.g., "100 billion words per day" in February 2024) and various disclosures. However, Epoch AI has done a tremendous service in [collating all of these](https://epoch.ai/data/ai-companies) in an easy to process format. 

Our goal is not only to find the number of tokens collectively processed each day today (Dec 2025), but to project this out into the future assuming exponential growth at current rates, so that we can find a crossover point. The assumption of exponential growth is very iffy in the long run, but it's the simplest model that we have that fits the data in the short run (the past 2-3 years).  

Every company has different types of data available for it, and different business models. For Anthropic in particular, we have barely *any* data beyond revenue and inference compute. We'll necessarily have to do a bit of guesswork. Consider these numbers in the spirit of a [Fermi estimate](https://en.wikipedia.org/wiki/Fermi_problem).

Nonetheless, even having Fermi-estimate tier numbers will tell us a lot about the scale of the AI welfare problem. 







## Summary


But even if some degree of AI model consciousness is possible, it raises the question of how we should compare AI welfare to human welfare. Most people agree that human consciousness is richer and deeper than the consciousness of a pig or chicken. Similarly, the first AIs to possess a degree of consciousness will likely be "less conscious" than a human would be.
