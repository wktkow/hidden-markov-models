# Hidden Markov Models

**Tomasz Burzykowski**
tomasz.burzykowski@uhasselt.be

---

## Earthquakes data

**Table 1.1** Number of major earthquakes (magnitude 7 or greater) in the world, 1900-2006; to be read across rows.

| | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 14 | 8 | 10 | 16 | 26 | 32 | 27 | 18 | 32 | 36 | 24 | 22 | 23 | 22 | 18 | 25 | 21 | 21 | 14 |
| 8 | 11 | 14 | 23 | 18 | 17 | 19 | 20 | 22 | 19 | 13 | 26 | 13 | 14 | 22 | 24 | 21 | 22 | 26 | 21 |
| 23 | 24 | 27 | 41 | 31 | 27 | 35 | 26 | 28 | 36 | 39 | 21 | 17 | 22 | 17 | 19 | 15 | 34 | 10 | 15 |
| 22 | 18 | 15 | 20 | 15 | 22 | 19 | 16 | 30 | 27 | 29 | 23 | 20 | 16 | 21 | 21 | 25 | 16 | 18 | 15 |
| 18 | 14 | 10 | 15 | 8 | 15 | 6 | 11 | 8 | 7 | 18 | 16 | 13 | 12 | 13 | 20 | 15 | 16 | 12 | 18 |
| 15 | 16 | 13 | 15 | 16 | 11 | 11 | | | | | | | | | | | | | |

<TextImageExplanation>
A line graph shows the count of major earthquakes per year from 1900 to 2006. The x-axis represents the year, ranging from 1900 to 2000 in increments of 20 years. The y-axis, labeled "count," ranges from 0 to 50 and represents the number of major earthquakes. The data points fluctuate significantly over time, with notable peaks around 1940 and several smaller peaks throughout the century.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Fitted Poisson
*   Overdispersion: sample mean 19.4, variance 51.6
*   Mixture?

<TextImageExplanation>
This is a plot representing the distribution of earthquake counts. The x-axis ranges from 0 to 40, representing the number of earthquakes. The y-axis shows the relative frequency or probability, ranging from 0.00 to 0.10. The plot consists of black dots representing observed data points and vertical lines that seem to represent a fitted distribution, which shows a poor fit to the data, suggesting that a simple Poisson model is inadequate.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Mixture: P(X=x) = ∑q P(Q=q)P(X=x | Q=q) = ∑q πq Pq(x)
*   Example: two Poisson rates, λ₁ and λ₂
*   E(X) = π₁λ₁ + (1-π₁)λ₂; Var(X) = E(X) + π₁(1-π₁)(λ₁-λ₂)²

<TextImageExplanation>
The image illustrates a two-component mixture model for earthquake data. It's divided into three columns: "active component," "component densities," and "observations." The "component densities" section at the top displays two overlapping bell-shaped curves, P1(x) and P2(x), representing the densities of the two components. The "active component" column on the left shows a sequence of observations where for each step, either component 1 (compt. 1) or component 2 (compt. 2) is active, indicated by a filled or empty circle. The "observations" column on the right lists the numerical value of the observation corresponding to each step.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Mixtures with 1 to 4 components

| model | i | δᵢ | λᵢ | -log L | mean | variance |
|---|---|---|---|---|---|---|
| m = 1 | 1 | 1.000 | 19.364 | 391.9189 | 19.364 | 19.364 |
| m = 2 | 1 | 0.676 | 15.777 | 360.3690 | 19.364 | 46.182 |
| | 2 | 0.324 | 26.840 | | | |
| m = 3 | 1 | 0.278 | 12.736 | 356.8489 | 19.364 | 51.170 |
| | 2 | 0.593 | 19.785 | | | |
| | 3 | 0.130 | 31.629 | | | |
| m = 4 | 1 | 0.093 | 10.584 | 356.7337 | 19.364 | 51.638 |
| | 2 | 0.354 | 15.528 | | | |
| | 3 | 0.437 | 20.969 | | | |
| | 4 | 0.116 | 32.079 | | | |
| **observations** | | | | | **19.364** | **51.573** |

<TextImageExplanation>
This slide displays four graphs and a table analyzing the earthquake data with mixture models. The four graphs show histograms of the earthquake counts, each overlaid with a fitted probability density curve for models with 1, 2, 3, and 4 components (labeled m=1 to m=4). The table on the left provides detailed parameters for each model, including the component weights (δᵢ), means (λᵢ), the negative log-likelihood (-log L), and the overall mean and variance of the fitted model, showing how the model's fit improves with more components.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Serial dependence
*   Mixtures assume independence

<TextImageExplanation>
This image shows an Autocorrelation Function (ACF) plot for the earthquake data. The horizontal axis is labeled "Lag" and ranges from 0 to 15. The vertical axis is labeled "ACF" and ranges from 0.0 to 1.0. Several vertical lines are plotted at integer lags, indicating the correlation of the time series with its past values; there are significant positive correlations at the first few lags, which gradually decrease as the lag increases.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Markov Chains

*   A sequence of discrete random variables (“states”)
    Q₁, Q₂, Q₃, ...
    *   We will assume a finite set of values 1,2, ..., m
*   In general:
    P(Q₁, Q₂,..., Qₜ₊₁) = P(Q₁)P(Q₂|Q₁)P(Q₃|Q₁,Q₂)...P(Qₜ₊₁|Q₁,...,Qₜ)
*   1st-order M. chain: P(Qₜ₊₁|Q₁, Q₂,..., Qₜ) ≡ P(Qₜ₊₁|Qₜ)
*   2nd-order M. chain: P(Qₜ₊₁|Q₁, Q₂,..., Qₜ) ≡ P(Qₜ₊₁|Qₜ₋₁,Qₜ)
*   Etc.

---

## Markov Chains

*   Transition probabilities: P(Qₛ₊ₜ = i | Qₛ = j )
*   Homogenous chain: no dependence on s
    [A(t)]ᵢⱼ = aᵢⱼ(t) = P(Qₛ₊ₜ = i | Qₛ = j )
*   It follows that A(t+u)=A(t)A(u) and A(t) = A(1)ᵗ
*   Hence, define A ≡ A(1), with aᵢⱼ ≡ aᵢⱼ(1)
*   Note that, for each row i, ∑ⱼ aᵢⱼ = 1

---

## Markov Chains

*   Consider u(t) = (P(Qₜ=1), P(Qₜ=2), ..., P(Qₜ=m))
    *   unconditional “state” distribution
*   u(1) – the initial “state” distribution
*   Note: u(t) = u(1)Aᵗ
*   A stationary distribution: u\*A = u\*
    *   If u(1) = u\*, then u(t) = u\*; the same distribution at all t
*   Stationary M. chain: u(t) the same for all t

---

## Earthquakes data

To deal with serial dependence in the data, assume that the Poisson rates depend on “states” forming a Markov chain.

<TextImageExplanation>
This diagram illustrates how a Hidden Markov Model can be applied to the earthquake data. The image is split into three columns: "Markov chain," "state-dependent distribution," and "observations." The "Markov chain" column shows a sequence of transitions between two states (represented by black and white circles) with associated probabilities. The "state-dependent distribution" column shows two distinct probability distributions, P1(x) and P2(x), corresponding to the two hidden states. The "observations" column on the right lists the numerical earthquake counts that are generated at each time step, dependent on the active hidden state.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## The “Fair Bet Casino”

*   The game is to flip coins, which results in only two possible outcomes: **Head** or **Tail**.
*   The **F**air coin will give **H**eads and **T**ails with same probability ½.
*   The **B**iased coin will give **H**eads with prob. ¾.

---

## The “Fair Bet Casino” (cont'd)

*   Thus, we define the probabilities:
    *   P(H|F) = P(T|F) = ½
    *   P(H|B) = ¾, P(T|B) = ¼
    *   The crooked dealer changes between Fair and Biased coins with probability 10%.

---

## The Fair Bet Casino Problem

*   **Input:** A sequence **x** = x₁x₂x₃...xₙ of coin tosses made by two possible coins (**F** or **B**).
*   **Output:** A sequence **q** = q₁q₂q₃...qₙ, with each qᵢ being either **F** or **B** indicating that xᵢ is the result of tossing the Fair or Biased coin respectively.

---

## Problem...

**Fair Bet Casino Problem**
Any observed outcome of coin tosses could have been generated by any sequence of states!
} -> Need to incorporate a way to grade different sequences differently. -> **Decoding Problem**

---

## P(x|fair coin) vs. P(x|biased coin)

*   Suppose first that dealer never changes coins. Some definitions...:
    *   **P(x|fair coin)**: prob. of the dealer using the F coin and generating the outcome x.
    *   **P(x|biased coin)**: prob. of the dealer using the B coin and generating outcome x.
    *   **k** the number of **H**eads in x.

---

## P(x|fair coin) vs. P(x|biased coin)

*   P(x|fair coin) = P(x₁...xₙ|fair coin) = Πᵢ₌₁,ₙ p(xᵢ|fair coin) = (1/2)ⁿ
*   P(x|biased coin) = P(x₁...xₙ|biased coin) = Πᵢ₌₁,ₙ p(xᵢ|biased coin) = (3/4)ᵏ(1/4)ⁿ⁻ᵏ = 3ᵏ/4ⁿ
*   **k** - the number of **H**eads in x.

---

## P(x|fair coin) vs. P(x|biased coin)

*   P(x |fair coin) = P(x |biased coin)
    when k = n / log₂3
    k ~ 0.67n

---

## Log-odds Ratio

We define **log-odds ratio** as follows:
log₂(P(x |fair coin) / P(x |biased coin))
= Σⁿᵢ₌₁ log₂(p(xᵢ | F) / p(xᵢ| B))
= n − k log₂3

*   Not really log-odds, but log-likelihood ratio
*   0 if k = n / log₂3
*   *Biased* (*fair*) coin most likely used if log-OR<0 (>0)

---

## Computing Log-odds Ratio in Sliding Windows

x₁x₂[x₃x₄x₅x₆x₇]x₈...xₙ

Consider a “sliding window” of the outcome sequence. Find the log-odds for this short window.

<TextImageExplanation>
The image illustrates the concept of a sliding window for analyzing a sequence of data. At the top, a sequence of variables x1, x2, ..., xn is shown, with a rectangular box highlighting a "sliding window" from x3 to x7. Below this, there is a horizontal line representing the "Log-odds value," with a central point at 0, which acts as a decision threshold. To the left of 0, it is labeled "Biased coin most likely used," and to the right, "Fair coin most likely used," indicating how the log-odds ratio is used for classification.
</TextImageExplanation>

**Disadvantages:**
- the length of the window is not known in advance
- different windows may classify the same position differently

---

## Hidden Markov Model (HMM)

*   Can be viewed as an abstract machine with *m* *hidden* states that emits symbols from an alphabet Σ with *r* symbols.
*   Each state has its own emission probability distribution.
*   The machine switches between states according to some probability distribution.
*   While in a certain state, the machine makes 2 decisions:
    *   What state should I move to next?
    *   What symbol - from the alphabet Σ - should I emit?

---

## Why “Hidden”?

*   Observers can see the emitted symbols of an HMM but have *no ability to know which state the HMM is currently in*.
*   Thus, the goal is to infer the most likely hidden states of an HMM based on the given sequence of emitted symbols.

---

## HMM Parameters

**Σ**: set of all *r* possible emission characters
Ex.: Σ = {H, T} for coin tossing
Σ = {1, 2, 3, 4, 5, 6} for dice tossing

**Q**: set of *m* hidden states, each emitting symbols from Σ
Q={F,B} for coin tossing

---

## HMM Parameters (cont'd)

**A = (aₖₗ)**: an *m x m* matrix of probability of changing from state *k* to state *l*
aFF = 0.9, aFB = 0.1
aBF = 0.1, aBB = 0.9

**E = (eₖ(b))**: an *m x r* matrix of probability of emitting symbol *b* during a step in which the HMM is in state *k*
eF(T) = ½, eF(H) = ½
eB(T) = ¼, eB(H) = ¾

---

## Markov Chain Property

P(q₁q₂q₃...qₙ) = P(q₁)P(q₂|q₁)P(q₃|q₂q₁)...P(qₙ|qₙ₋₁...q₂q₁)
≡ P(q₁)P(q₂|q₁)P(q₃|q₂)...P(qₙ|qₙ₋₁)
= P(q₁) a(q₁,q₂) a(q₂,q₃) ... a(qₙ₋₁,qₙ)

**Additionally:**
Given the state, the emission of different symbols is independent.
The emission of different symbols in different states is independent.

---

## HMM for Fair Bet Casino

*   The *Fair Bet Casino* in HMM terms:
    Σ = {0, 1} (0 for **T**ails and 1 **H**eads)
    Q = {F,B} – F for Fair & B for Biased coin.
*   Transition Probabilities A

| | Biased | Fair |
|---|---|---|
| **Biased** | aBB = 0.9 | aFB = 0.1 |
| **Fair** | aBF = 0.1 | aFF = 0.9 |

---

## HMM for Fair Bet Casino (cont'd)

Emission Probabilities E

| | Tails(0) | Heads(1) |
|---|---|---|
| **Fair** | eF(0) = ½ | eF(1) = ½ |
| **Biased** | eB(0) = ¼ | eB(1) = ¾ |

---

## HMM for Fair Bet Casino (cont'd)

<TextImageExplanation>
This image is a state diagram representing the Hidden Markov Model for the "Fair Bet Casino" problem. There are two hidden states, F (Fair) and B (Biased), shown in squares. Arrows indicate transitions between states: loops on F and B have a probability of 9/10 (0.9), while arrows between F and B have a probability of 1/10 (0.1). From each state, dashed arrows point to the possible emissions (H for Heads, T for Tails) in dashed circles, with their respective emission probabilities labeled (e.g., from F, both H and T have probability 1/2; from B, H has 3/4 and T has 1/4).
</TextImageExplanation>

HMM model for the *Fair Bet Casino* Problem

---

## Hidden Paths

*   A *path* **q** = q₁...qₙ in the HMM is defined as a sequence of states.
*   Consider path **q** = FFFBBBBBFFF and sequence **x** = 01011101001

<TextImageExplanation>
This slide illustrates the concept of a hidden path in an HMM with a specific example. Two rows of data are presented: the top row 'x' is the observed sequence of 0s and 1s, and the row below 'q' is the corresponding hidden path of states (F or B). Below these, two more rows show the emission probabilities P(xi|qi) and transition probabilities P(qi-1 -> qi) for each step in the sequence. An arrow points from the hidden state qi to the observed symbol xi, explaining that P(xi|qi) is the probability that xi was emitted from state qi. Another arrow points from one state to the next, explaining that P(qi-1 -> qi) is the transition probability from state qi-1 to state qi.
</TextImageExplanation>

---

## P(x) Calculation

*   P(x): Probability of observing sequence x, given the model M.
    P(x) = Σq P(x|q) · P(q)
    = Σq{P(q₀→q₁)·P(x₁| q₁)·P(q₁ → q₂)·...·P(qₙ₋₁ → qₙ)·P(xₙ| qₙ)}
    = Σq{a(q₀,q₁)· e(q₁)(x₁)·a(q₁,q₂)·...· a(qₙ₋₁,qₙ)·e(qₙ)(xₙ)}
    = Σq{a(q₀,q₁) Πⁿₜ₌₁ e(qₜ)(xₜ) · Πⁿ⁻¹ₜ₌₁ a(qₜ,qₜ₊₁)}
*   P(q₀→q₁) = a(q₀,q₁): for starting in state q₁ (imaginary state 0 “start”)
*   Requires 2nmⁿ computations
    *   Sum of mⁿ terms, each being a product with 2n multiplications
    *   Impossible numerically: for m=5 states, n=100, 2·100·5¹⁰⁰ ≈ 10⁷²

---

## P(x) Calculation: Forward Algorithm

*   One can write
    P(x) = Σᵐᵢ₌₁ P(x, qₙ=Qᵢ)
*   “Forward variable”: P(x₁, ..., xₜ, qₜ=Qᵢ)
*   The following holds
    P(x₁, q₁=Qᵢ) = e(Qᵢ)(x₁) a(q₀,Qᵢ)
    P(x₁, ..., xₜ₊₁, qₜ₊₁=Qᵢ) = {Σᵐⱼ₌₁ P(x₁, ..., xₜ, qₜ=Qⱼ) a(Qⱼ,Qᵢ) } e(Qᵢ)(xₜ₊₁)
*   Recursion!
*   Requires ~ nm² computations
    *   m values (t=1,2,...), each a sum of m products
    *   Feasible numerically: for m=5 states, n=100, 25·100 ≈ 2500, not 10⁷²

---


---

## P(x) Calculation: Backward Algorithm

*   One can also write
    P(x) = Σᵐᵢ₌₁ P(x, qₜ=Qᵢ) =
    Σᵢ P(x₁,...,xₜ, qₜ=Qᵢ) P(xₜ₊₁,...,xₙ|x₁,...,xₜ, qₜ=Qᵢ) =
    Σᵢ P(x₁,...,xₜ, qₜ=Qᵢ) P(xₜ₊₁,...,xₙ| qₜ=Qᵢ)

*   P(x₁,...,xₜ, qₜ=Qᵢ) come from the forward algorithm

---

## Backward Algorithm

*   “Backward variable”: P(xₜ₊₁,...,xₙ| qₜ=Qⱼ)

*   The following holds
    P(xₙ|qₙ₋₁=Qᵢ) = Σᵐⱼ₌₁ a(Qᵢ,Qⱼ) e(Qⱼ)(xₙ)
    P(xₜ,...,xₙ|qₜ₋₁=Qᵢ) = {Σᵐⱼ₌₁ P(xₜ₊₁,...,xₙ|qₜ=Qⱼ) a(Qᵢ,Qⱼ)} e(Qᵢ)(xₜ)

*   Recursion again!

---

## P(x) Calculation

*   We can use matrix notation:
    P(x) = Σq{a(q₀,q₁)· e(q₁)(x₁)·a(q₁,q₂)·...· a(qₙ₋₁,qₙ)·e(qₙ)(xₙ)}
    = a₀E(x₁)AE(x₂) ·...· AE(xₙ)1ᵀ
    where a₀ is the initial state distribution, A is the transition probability matrix, 1 is the vector of ones, and E(x)=diag(p₁(x), p₂(x),...,pₘ(x)).

*   Let Bₜ = AE(xₜ), then P(x) = a₀E(x₁)B₂·...·Bₙ1ᵀ

*   If a₀ is the stationary distribution, then a₀A=a₀, so
    P(x) = a₀E(x₁)B₂·...·Bₙ1ᵀ = a₀AE(x₁)B₂·...·Bₙ1ᵀ = a₀B₁B₂·...·Bₙ1ᵀ

---

## P(x) Calculation

*   In matrix notation, P(x) = a₀E(x₁)AE(x₂)·...·AE(xₙ)1ᵀ

*   Define αₜ = a₀E(x₁)AE(x₂)·...·AE(xₜ) = a₀E(x₁) Πᵗₛ₌₂AE(xₛ)

*   P(x) = αₙ1ᵀ

*   And αₜ = αₜ₋₁AE(xₜ) for t>1, with α₁ = a₀E(x₁). Hence, recursion.
    *   Note: elements of αₜ are forward probabilities.

---

## “Optimal” State Sequence?

**Given:** a sequence of symbols generated by an HMM.

**Goal:** find the path of states most likely to generate the observed sequence.

---

## Individually Most Likely States (Local Decoding)

*   For each t, we may look for maxᵢ P(qₜ=Qᵢ | x)

    P(qₜ=Qᵢ | x) = P(x, qₜ=Qᵢ) / P(x) =
    [P(x₁, ..., xₜ, qₜ=Qᵢ) P(xₜ₊₁, ..., xₙ|qₜ=Qᵢ)] / [Σⱼ P(x₁, ..., xₜ, qₜ=Qⱼ) P(xₜ₊₁, ..., xₙ|qₜ=Qⱼ)]

*   Thus, we can use forward-backward algorithms

---

## Individually Most Likely States (cont'd)

*   The resulting sequence of states can be problematic.

*   This is because the states are optimized *individually*, without regard of the probability of occurrence of a *sequence* of states.

*   Alternative: find a path that maximizes P(q|x) over all possible paths q.

---

## Global Decoding Problem

*   **Goal:** Find an optimal hidden path of states given observations.

*   **Input:** Sequence of observations x = x₁...xₙ generated by an HMM M(Σ, Q, A, E)

*   **Output:** A path that maximizes P(q | x) over all possible paths q.

---

## Building Manhattan for Global Decoding Problem

*   Andrew Viterbi used the Manhattan grid model to solve the *Decoding Problem*.
*   Every choice of q corresponds to a path in the graph
*   The only valid direction in the graph is *eastward*.
*   This graph has m²(n-1) edges, each with a weight.

<TextImageExplanation>
This image displays a trellis diagram, a type of graph used to visualize the possible state sequences in a Hidden Markov Model. The graph is organized into a grid with `m states` arranged vertically and `n layers` (time steps) arranged horizontally. Each node in a given layer is connected by a directed arrow to every node in the subsequent layer, representing all possible transitions between states from one time step to the next.
</TextImageExplanation>

---

## Decoding Problem as Finding a Longest Path in a DAG

*   The *Decoding Problem* is reduced to finding a longest (largest score) path in the *directed acyclic graph* (DAG) above.

*   **Notes:** the length of the path is defined as the *product* of its edges' weights.

---

## Global Decoding Problem (cont'd)

*   Idea: max_q P(q|x) = max_q P(q,x)/P(x) = max_q P(q,x)
*   So, argmax_q P(q|x) = argmax_q P(q,x)
*   Every path in the graph has the probability P(q,x)
*   The Viterbi algorithm finds the path that maximizes P(q,x) among all possible paths
    *   It runs in *O(nm²)* time.

---

## Global Decoding Problem (cont'd)

P(x|q) = a(q₀,q₁) · Π {e(qᵢ)(xᵢ) · a(qᵢ,qᵢ₊₁)}

<TextImageExplanation>
The image shows a simple directed graph representing a single state transition in an HMM. There are two nodes, one labeled `(k, i)` representing state `k` at time `i`, and another labeled `(l, i+1)` representing state `l` at time `i+1`. A directed edge with weight `w` connects the first node to the second, symbolizing the transition.
</TextImageExplanation>

The weight **w** is given by:
w = eₗ(xᵢ₊₁) · aₖₗ

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
This is a trellis diagram representing the state space of an HMM over time, used for the decoding problem. The graph has `m` rows of nodes representing the `m states` and `n` columns representing `n layers` or time steps. Each node in the first column is labeled, from `s₁,₁` at the top to `sₘ,₁` at the bottom, indicating the initial state probabilities.
</TextImageExplanation>

*   **Initialization:**
    sₖ,₁ = eₖ(x₁) · a(begin,k)

---

## Decoding Problem and Dynamic Programming

sₗ,ᵢ₊₁ = maxₖ ∈ Q{sₖ,ᵢ · weight of edge between (k,i) and (l,i+1)} =
maxₖ ∈ Q{sₖ,ᵢ · aₖₗ · eₗ(xᵢ₊₁)} =
eₗ(xᵢ₊₁) · maxₖ ∈ Q{sₖ,ᵢ · aₖₗ}

<TextImageExplanation>
This diagram illustrates the core recursive step of the Viterbi algorithm. On the left, at time `t`, there are `N` states labeled `s₁` to `sN`. All these states are connected to a single state `sj` at the next time step, `t+1`. This visualization shows how the score (or probability) of being in state `sj` at time `t+1` is calculated by considering all possible paths from the previous states at time `t` and selecting the one with the maximum score.
</TextImageExplanation>

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
This slide combines the full trellis diagram of the HMM on the left with a smaller diagram on the right that illustrates the dynamic programming step. The large trellis shows the entire space of possible state sequences over `n` time steps. The smaller diagram highlights the calculation for a single node `sj` at time `t+1`, which depends on the scores of all nodes at the previous time `t`, visually connecting the abstract algorithm to its application on the graph.
</TextImageExplanation>

sₗ,ᵢ₊₁ = maxₖ ∈ Q{sₖ,ᵢ · weight of edge between (k,i) and (l,i+1)} =
maxₖ ∈ Q{sₖ,ᵢ · aₖₗ · eₗ(xᵢ₊₁)} =
eₗ(xᵢ₊₁) · maxₖ ∈ Q{sₖ,ᵢ · aₖₗ}

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
This image shows the trellis diagram and highlights the progression of the Viterbi algorithm from the first to the second time step. A red arrow points from the first node in the top row to the second node, labeled with the calculation `e₁(x₂)·maxₖ{sₖ,₁·aₖ₁}`. Similarly, a red arrow points from the nodes in the first layer to the bottom node in the second layer, labeled `eₘ(x₂)·maxₖ{sₖ,₁·aₖₘ}`, demonstrating how the scores for each state in the second layer are computed.
</TextImageExplanation>

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
The image displays the full trellis diagram, which visualizes all possible paths through the hidden states of an HMM over time. The nodes in the first column are labeled `s₁,₁` to `sₘ,₁`, representing the initial scores for each state. The nodes in the final column are labeled `s₁,ₙ` to `sₘ,ₙ`, representing the final scores of the most probable paths ending in each respective state after `n` steps.
</TextImageExplanation>

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
This image illustrates the termination step of the Viterbi algorithm using the trellis diagram. After the forward pass is complete and all node scores are calculated up to the last layer `n`, a red arrow points upward towards the final column of nodes. The text "Find max" below the arrow indicates that the algorithm now finds the node with the highest score in this final layer, which identifies the end state of the most likely hidden path.
</TextImageExplanation>

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
The image displays the trellis diagram after the maximization step of the Viterbi algorithm. The node `s₂,ₙ` is highlighted in red, indicating that it has the highest score among all nodes in the final layer (`n`). This means the most probable sequence of hidden states is the one that ends in state 2 at the final time step.
</TextImageExplanation>

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
This image illustrates the beginning of the traceback phase in the Viterbi algorithm. The final node of the optimal path, `s₂,ₙ`, is highlighted in red. The formula "Find maxₖ sₙ₋₁,ₖ·aₖ,₂" is shown below, explaining that the next step is to find which state `k` at time `n-1` most likely transitioned to state 2 at time `n`.
</TextImageExplanation>

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
This image shows a step in the Viterbi algorithm's traceback process. A red arrow points backwards from the identified final state of the optimal path, `s₂,ₙ`, to the state in the previous layer (`n-1`) that most likely preceded it. This demonstrates how the algorithm reconstructs the most probable path by tracing back from the end.
</TextImageExplanation>

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
The image continues to illustrate the traceback phase of the Viterbi algorithm on the trellis diagram. A red arrow points from the final optimal state `s₂,ₙ` to its predecessor in layer `n-1`. The formula `Find maxₖ sₙ₋₂,ₖ·aₖ,₁` is shown below, indicating the process continues by finding the most likely predecessor in layer `n-2` for the now-identified state in layer `n-1`.
</TextImageExplanation>

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
This image shows the Viterbi traceback process in action. Two red arrows now trace the optimal path backward from the final state `s₂,ₙ` through two time steps. This visualization clearly shows how the most likely sequence of hidden states is reconstructed step-by-step, moving from the end of the sequence toward the beginning.
</TextImageExplanation>

---

## Edit Graph for Decoding Problem

<TextImageExplanation>
This image displays the fully reconstructed optimal path found by the Viterbi algorithm, highlighted in red on the trellis diagram. The path represents the single most likely sequence of hidden states that could have generated the observed data. This final highlighted path is the output of the decoding problem.
</TextImageExplanation>

---

## Viterbi Algorithm

*   The value of the product can become extremely small, which leads to overflowing.
*   To avoid overflowing, use log value instead.
    s'ₗ,ᵢ₊₁ = log eₗ(xᵢ₊₁) + maxₖ ∈ Q{s'ₖ,ᵢ + log(aₖₗ)}
*   Note similarity to forward algorithm:
    P(x₁, ..., xₜ₊₁, qₜ₊₁=Qₖ) = e(Qₖ)(xₜ₊₁) · Σⱼ P(x₁, ..., xₜ, qₜ=Qⱼ) a(Qⱼ,Qₖ)
    sₗ,ᵢ₊₁ = eₗ(xᵢ₊₁) · maxₖ ∈ Q{sₖ,ᵢ · aₖₗ}

---

## State Prediction

*   Consider P(qₜ=Qᵢ | x)
*   We get
    1≤t<n: P(x₁,...,xₜ,qₜ=Qᵢ)P(xₜ₊₁,...,xₙ|qₜ=Qᵢ)/P(x)
    t=n: P(x₁,...,xₙ,qₙ=Qᵢ) / P(x)
    t>n: αₙ[Aᵗ⁻ⁿ](.,i)/P(x)
*   For t>n, we can write P(qₙ₊ₕ=Qᵢ|x) = αₙ[Aʰ](.,i)/P(x)
    *   with h → +∞, αₙAʰ/P(x) → stationary distribution

---

## HMM Parameter Estimation

*   So far, we have assumed that the transition and emission probabilities are known.
*   However, in most HMM applications, the probabilities are not known. It's very hard to estimate the probabilities.

---

## HMM Parameter Estimation (cont'd)

*   Let Θ be a vector combining the unknown transition and emission probabilities.
*   Given training sequences x¹,...,xᴹ, let P(x|Θ) be the prob. of x given the assignment of parameters Θ.
*   Then our goal is to find MLE:
    max_Θ Πᴹⱼ₌₁ P(xʲ|Θ)

---

## HMM Parameter Estimation (cont'd)

*   Issue: constrained parametrization, as A1ᵀ=1ᵀ
    *   row entries sum to one
*   Consider “working” parameters ϑᵢⱼ for i ≠ j
*   Let aᵢⱼ=exp(ϑᵢⱼ)/{1+Σₖexp(ϑᵢₖ)} for i ≠ j and aᵢᵢ= 1/{1+Σₖexp(ϑᵢₖ)}
*   Additional contraints for, e.g., emission-distribution parameters can be handled by transformations
    *   Poisson rates: use ln(λ)

---

## HMM Parameter Estimation (cont'd)

*   In matrix notation, P(x) = a₀E(x₁)AE(x₂)·...·AE(xₙ)1ᵀ = αₙ1ᵀ
    with α₁ = a₀E(x₁) and recursion αₜ = αₜ₋₁AE(xₜ) for t>1.
*   Hence, one could directly maximize Πᴹⱼ₌₁ P(xʲ|Θ)
*   Issue: numerical underflow
*   Solution: log-transformation and scaling:
    for stationary a₀, ln P(x) = Σⁿₜ₌₁ ln[{αₜ₋₁/(αₜ₋₁1ᵀ)}AE(xₜ)1ᵀ]
    for non-stationary, ln{a₀E(x₁)1ᵀ} + Σⁿₜ₌₂ ln[{αₜ₋₁/(αₜ₋₁1ᵀ)}AE(xₜ)1ᵀ]

---

## HMM Parameter Estimation (cont'd)

*   Assume state paths are known for the training set.
*   Maximum likelihood estimators for aᵢⱼ and eᵢ(s) are
    aᵢⱼ = Aᵢⱼ / Σⱼ' Aᵢⱼ' and eᵢ = Eᵢ(s) / Σs' Eᵢ(s')
    where Aᵢⱼ and Eᵢ(s) are the numbers of times a particular transition and emission are used in training sequences.

---

## Baum-Welch Algorithm

*   Assume further that state paths are not known.
*   B-W is a version of the E-M algorithm.
    *   **E-step:** Aᵢⱼ and Eᵢ(s) are estimated using current estimates of aᵢⱼ and eᵢ(s).
    *   **M-step:** Using the estimates, aᵢⱼ and eᵢ(s) are updated:
        aᵢⱼ = Aᵢⱼ / Σⱼ' Aᵢⱼ' and eᵢ = Eᵢ(s) / Σs' Eᵢ(s')

---

## Baum-Welch Algorithm (cont'd)

*   The probability that transition Qᵢ → Qⱼ is used at position t in x is
    P(qₜ=Qᵢ, qₜ₊₁=Qⱼ | x) =
    P(x₁, ..., xₜ, qₜ=Qᵢ)aᵢⱼeⱼ(xₜ₊₁)P(xₜ₊₂, ..., xₙ | qₜ₊₁=Qⱼ) / P(x)
*   The expected number of times transition Qᵢ → Qⱼ is used in the training sequences is
    Aᵢⱼ = Σᴹₖ₌₁ Σ P(xᵏ₁, ..., xᵏₜ, qₜ=Qᵢ)aᵢⱼeⱼ(xᵏₜ₊₁)P(xᵏₜ₊₂, ..., xₙᵏ | qₜ₊₁=Qⱼ) / P(xᵏ)

<TextImageExplanation>
This diagram illustrates the computation required for the E-step of the Baum-Welch algorithm. It shows a segment of a trellis with a transition from state Si at time t to state Sj at time t+1. The calculation involves the forward probability αt(i), the transition probability aij, the emission probability bj(Ot+1), and the backward probability βt+1(j) to compute the probability of that specific transition given the entire observation sequence.
</TextImageExplanation>

---

## Baum-Welch Algorithm (cont'd)

*   Similarly, the expected number of times letter s is emitted in state Qᵢ is
    Eᵢ(s) = Σₖ Σₜ' P(xᵏ₁, ..., xᵏₜ', qₜ'=Qᵢ)P(xᵏₜ'₊₁, ..., xₙᵏ| qₜ'=Qᵢ)/P(xᵏ)
    where the sum is over positions t', at which s was observed.
*   The expected number of times sequence starts in state Qᵢ is
    a(begin,i) = Σₖ P(xᵏ₁, q₁=Qᵢ)P(xᵏ₂, ..., xₙᵏ| q₁=Qᵢ)/P(xᵏ)

---

## Baum-Welch Algorithm (cont'd)

*   Initialize aᵢⱼ and eᵢ(s), Aᵢⱼ, and Eᵢ(s).
*   **E-step:**
    *   for each training sequence k
        *   use the forward algorithm to compute P(xᵏ₁, ..., xᵏₜ, qₜ=Qᵢ)
        *   use the backward algorithm to compute P(xᵏₜ, ..., xₙᵏ|qₜ=Qⱼ)
    *   calculate the expected values Aᵢⱼ and Eᵢ(s)
*   **M-step:** compute the updated values of aᵢⱼ and eᵢ(s)
*   Iterate until convergence

---

## Viterbi Training

*   A modification of the B-W algorithm.
*   Given aᵢⱼ and eᵢ(s), most likely paths are found for the training sequences, and used to re-compute Aᵢⱼ and Eᵢ(s).
*   B-W maximizes P(x¹, ..., xᴹ|Θ)
*   Viterbi training maximizes P(x¹, ..., xᴹ | Θ, q(x¹), ..., q(xᴹ))

---

## Baum-Welch/Viterbi Training (cont'd)

*   **Caution:**
    If there is a small group of sequences in the training set which are highly similar, the model will overspecialize to the small group
    => use a method of sequence weighting

---

## HMM Estimation Issues

*   Local maxima can occur
*   Use various starting values and check stability of solutions
*   For the emission-parameters, use observed quantiles
    *   Poisson HMM: if 3 states, use quartiles of the count distribution
*   Transition probabilities: less trivial
    *   Uniform off-diagonal probabilities (all 0.01)?

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Direct (stationary) likelihood maximization

| | mean | variance |
|---|---|---|
| observations: | 19.364 | 51.573 |
| 'one-state HMM': | 19.364 | 19.364 |
| two-state HMM: | 19.086 | 44.523 |
| three-state HMM: | 18.322 | 50.709 |
| four-state HMM: | 18.021 | 49.837 |

<TextImageExplanation>
This slide presents results of fitting HMMs with different numbers of states to earthquake data. It includes two plots for a 2-state model (m=2) and two for a 3-state model (m=3). For each model, one plot shows the histogram of the data with the fitted mixture density overlaid, and the second plot shows the time series of earthquake counts with horizontal lines indicating the means of the hidden states, illustrating how the model segments the data.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   EM

**Table 4.1** Two-state model for earthquakes, fitted by EM.
| Iteration | γ₁₂ | γ₂₁ | λ₁ | λ₂ | δ₁ | -l |
|---|---|---|---|---|---|---|
| 0 | 0.100000 | 0.10000 | 10.000 | 30.000 | 0.50000 | 413.27542 |
| 1 | 0.138816 | 0.11622 | 13.742 | 24.169 | 0.99963 | 343.76023 |
| 2 | 0.115510 | 0.10079 | 14.090 | 24.061 | 1.00000 | 343.13618 |
| 30 | 0.071653 | 0.11895 | 15.419 | 26.014 | 1.00000 | 341.87871 |
| 50 | 0.071626 | 0.11903 | 15.421 | 26.018 | 1.00000 | 341.87870 |
| convergence | 0.071626 | 0.11903 | 15.421 | 26.018 | 1.00000 | 341.87870 |
| **stationary model** | 0.065961 | 0.12851 | 15.472 | 26.125 | 0.66082 | 342.31827 |

**Table 4.2** Three-state model for earthquakes, fitted by EM.
| Iteration | λ₁ | λ₂ | λ₃ | δ₁ | δ₂ | -l |
|---|---|---|---|---|---|---|
| 0 | 10.000 | 20.000 | 30.000 | 0.33333 | 0.33333 | 342.90781 |
| 1 | 11.699 | 19.030 | 29.741 | 0.92471 | 0.07487 | 332.12143 |
| 2 | 12.265 | 19.078 | 29.581 | 0.99588 | 0.00412 | 330.63689 |
| 30 | 13.134 | 19.713 | 29.710 | 1.00000 | 0.00000 | 328.52748 |
| convergence | 13.134 | 19.713 | 29.710 | 1.00000 | 0.00000 | 328.52748 |
| **stationary model** | 13.146 | 19.721 | 29.714 | 0.44364 | 0.40450 | 329.46028 |

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   EM
    *   Three-state model with initial distribution (1,0,0), fitted by EM:
        Γ = [[0.9393, 0.0321, 0.0286], [0.0404, 0.9064, 0.0532], [0.0000, 0.1903, 0.8097]]
        λ = (13.134, 19.713, 29.710).

    *   Three-state model based on stationary Markov chain, fitted by direct numerical maximization:
        Γ = [[0.9546, 0.0244, 0.0209], [0.0498, 0.8994, 0.0509], [0.0000, 0.1966, 0.8034]]
        δ = (0.4436, 0.4045, 0.1519),
        and λ = (13.146, 19.721, 29.714).

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Most likely states (non-stationary)

<TextImageExplanation>
This slide displays several bar charts visualizing the probabilities of being in different hidden states over time for the earthquake data. On the left, there are three plots labeled "state 1", "state 2", and "state 3", showing the decoded state probabilities for a 3-state model from 1900 to 2000. On the right, four plots labeled "state 1" through "state 4" show the decoded state probabilities for a 4-state model over the same time period. These plots reveal how the model assigns different periods of seismic activity to different underlying states.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Local decoding
*   Four-state HMM “splits” the “lowest” state
    *   the state visited only for 1919-1922 & 1981-1989

<TextImageExplanation>
This slide presents two time-series plots of the earthquake data, comparing a 3-state model (top, m=3) with a 4-state model (bottom, m=4). Both plots show the annual earthquake counts from 1900 to 2000, with horizontal lines representing the means of the decoded hidden states. The comparison illustrates how the 4-state model further refines the segmentation of the data by splitting the lowest-level state of the 3-state model into two distinct, lower-activity states.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Global decoding (Viterbi)
*   Four-state HMM “splits” the “lowest” state
    *   the state visited only for 1919-1922 & 1981-1989

<TextImageExplanation>
This slide is nearly identical to the previous one, showing two time-series plots for 3-state (m=3) and 4-state (m=4) models. The plots display annual earthquake counts with horizontal lines indicating the means of the hidden states as determined by Viterbi (global) decoding. The purpose is to show the state segmentation produced by the Viterbi algorithm and how the 4-state model refines the results of the 3-state model.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Local and global (Viterbi) decoding
*   Difference: 1911 & 1941

<TextImageExplanation>
This slide compares the results of local and global (Viterbi) decoding for 3-state (m=3, top row) and 4-state (m=4, bottom row) HMMs applied to the earthquake data. Each of the four plots shows the time series of earthquake counts with decoded state means. Pink circles on the top two plots (m=3) highlight the years 1911 and 1941, indicating specific points where the local and global decoding methods produce different state assignments.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   State predictions

| year | 2007 | 2008 | 2009 | 2016 | 2026 | 2036 |
|---|---|---|---|---|---|---|
| state=1 | 0.951 | 0.909 | 0.871 | 0.674 | 0.538 | 0.482 |
| 2 | 0.028 | 0.053 | 0.077 | 0.220 | 0.328 | 0.373 |
| 3 | 0.021 | 0.038 | 0.052 | 0.107 | 0.134 | 0.145 |

**Table 4.2** Three-state model for earthquakes, fitted by EM.
| Iteration | λ₁ | λ₂ | λ₃ | δ₁ | δ₂ | -l |
|---|---|---|---|---|---|---|
| 0 | 10.000 | 20.000 | 30.000 | 0.33333 | 0.33333 | 342.90781 |
| 1 | 11.699 | 19.030 | 29.741 | 0.92471 | 0.07487 | 332.12143 |
| 2 | 12.265 | 19.078 | 29.581 | 0.99588 | 0.00412 | 330.63689 |
| 30 | 13.134 | 19.713 | 29.710 | 1.00000 | 0.00000 | 328.52748 |
| convergence | 13.134 | 19.713 | 29.710 | 1.00000 | 0.00000 | 328.52748 |
| **stationary model** | 13.146 | 19.721 | 29.714 | 0.44364 | 0.40450 | 329.46028 |

**REF: Zucchini & MacDonald (2009)**

---

## HMM Selection

*   Number of states?
*   AIC or BIC

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Model selection

<TextImageExplanation>
This slide addresses model selection for the earthquake data. A plot shows AIC and BIC values for HMMs with 1 to 6 states, with the y-axis representing the criterion value and the x-axis the number of states; both criteria suggest a 3-state model is optimal. Below the plot, a table provides the specific values for the number of parameters (k), negative log-likelihood (-log L), AIC, and BIC for HMMs and independent mixture models with varying numbers of components, reinforcing that the 3-state HMM has the lowest AIC and BIC values.
</TextImageExplanation>

**Not a reasonable choice**
Multimodal likelihood <-- 5-state HM

**REF: Zucchini & MacDonald (2009)**

---

## HMM Diagnostics

*   ***Uniform pseudo-residuals:*** rₜ = P(Xₜ ≤ xₜ) = Fₜ(xₜ) ~ U(0,1)
    *   If the model Xₜ ~ Fₜ, where Xₜ continuous, is correct
    *   Not easy to detect outliers (0.97 and 0.999)
*   ***Normal pseudo-residuals:*** zₜ = Φ⁻¹{Fₜ(xₜ)} ~ N(0,1)
    *   zₜ = 0 if xₜ = median
    *   Easier to detect outliers

<TextImageExplanation>
This slide explains how pseudo-residuals can be used for HMM diagnostics. A series of diagrams illustrates the transformation of data from its original distribution to a standard normal distribution. For each observation xt, it is first mapped to a uniform distribution via its cumulative distribution function (CDF), uₜ=Fₜ(xₜ), and then transformed to a standard normal variable, zₜ=Φ⁻¹(uₜ). This process is shown for three different points in a time series (x₁, x₂, xT), demonstrating how diagnostics can be performed on the transformed residuals.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## HMM Diagnostics

*   ***Uniform pseudo-residual segments*** for discrete Xₜ:
    [rₜ⁻, rₜ⁺] = [Fₜ(xₜ⁻), Fₜ(xₜ)]
    *   xₜ⁻ - largest realization of Xₜ strictly less than xₜ
*   ***Normal pseudo-residual segments*** for discrete Xₜ:
    [zₜ⁻, zₜ⁺] = [Φ⁻¹(rₜ⁻), Φ⁻¹(rₜ⁺)]
*   For a Q-Q plot, order zₜ\* = Φ⁻¹{(rₜ⁻+rₜ⁺)/2}
    *   zₜ\* can be used for diagnostics as well

<TextImageExplanation>
This slide illustrates the concept of pseudo-residual segments for discrete data in HMM diagnostics. On the left, a bar chart shows a discrete probability distribution Pₜ(x) with probabilities for values less than, equal to, and greater than a specific observation xₜ. On the right, this discrete distribution is mapped onto a standard normal curve, where the single point xₜ is transformed into an interval [zₜ⁻, zₜ⁺], representing the range of normal quantiles corresponding to the cumulative probability up to and including xₜ.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## HMM Diagnostics

*   ***Ordinary pseudo-residuals:*** zₜ = Φ⁻¹{P(Xₜ ≤ xₜ| x₁,..., xₜ₋₁, xₜ₊₁,...,xₙ)}
    *   If the HMM is correct, zₜ should be N(0,1) distributed
*   ***Ordinary pseudo-residual segments:*** [zₜ⁻, zₜ⁺] with
    zₜ⁻ = Φ⁻¹{P(Xₜ < xₜ| x₁,..., xₜ₋₁, xₜ₊₁,...,xₙ)} and
    zₜ⁺ = Φ⁻¹{P(Xₜ ≤ xₜ| x₁,..., xₜ₋₁, xₜ₊₁,...,xₙ)}
*   Note that
    P(Xₜ=x | x₁,..., xₜ₋₁, xₜ₊₁,...,xₙ) = P(x₁,..., x,...,xₙ) / P(x₁,..., xₜ₋₁, xₜ₊₁,...,xₙ) =
    a₀E(x₁)B₂...Bₜ₋₁AE(x)Bₜ₊₁...Bₙ1ᵀ / a₀E(x₁)B₂...Bₜ₋₁ABₜ₊₁...Bₙ1ᵀ
    ∝ αₜ₋₁AE(x)Bₜ₊₁...Bₙ1ᵀ

**REF: Zucchini & MacDonald (2009)**

---

## HMM Diagnostics

*   ***Forecast pseudo-residuals:*** zₜ = Φ⁻¹{P(Xₜ ≤ xₜ| x₁,..., xₜ₋₁)}
    *   Deviation from the median of the one-step-ahead forecast
    *   If indicating an outlier, unacceptable description of the series by the HMM
    *   Possible monitoring of a series
*   ***Forecast pseudo-residual segments:*** [zₜ⁻, zₜ⁺] with
    zₜ⁻ = Φ⁻¹{P(Xₜ < xₜ| x₁,..., xₜ₋₁)} and zₜ⁺ = Φ⁻¹{P(Xₜ ≤ xₜ| x₁,..., xₜ₋₁)}
*   Note that
    P(Xₜ=xₜ| x₁,..., xₜ₋₁) = P(x₁,..., xₜ) / P(x₁,..., xₜ₋₁) =
    a₀E(x₁)B₂...Bₜ₋₁AE(x)1ᵀ / a₀E(x₁)B₂...Bₜ₋₂AE(x)1ᵀ
    = αₜ₋₁AE(x)1ᵀ / αₜ₋₁1ᵀ

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Autocorrelation functions
    *   two states: ρ(k) = 0.5713 × 0.8055ᵏ;
    *   three states: ρ(k) = 0.4447 × 0.9141ᵏ + 0.1940 × 0.7433ᵏ;
    *   four states: ρ(k) = 0.2332 × 0.9519ᵏ + 0.3682 × 0.8174ᵏ + 0.0369 × 0.7252ᵏ.

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| observations | 0.570 | 0.444 | 0.426 | 0.379 | 0.297 | 0.251 | 0.251 | 0.149 |
| 2-state model | 0.460 | 0.371 | 0.299 | 0.241 | 0.194 | 0.156 | 0.126 | 0.101 |
| 3-state model | 0.551 | 0.479 | 0.419 | 0.370 | 0.328 | 0.292 | 0.261 | 0.235 |
| 4-state model | 0.550 | 0.477 | 0.416 | 0.366 | 0.324 | 0.289 | 0.259 | 0.234 |

<TextImageExplanation>
This slide presents a comparison of the sample Autocorrelation Function (ACF) with the theoretical ACFs from fitted HMMs with two, three, and four states. The bar chart shows the ACF for lags 0 to 15, where the bold bars on the far left of each group represent the sample ACF from the data. The subsequent thinner bars represent the ACFs of the two, three, and four-state models, respectively, illustrating how well each model captures the serial dependence in the earthquake data.
</TextImageExplanation>

**Figure 6.2 Earthquakes data:** sample ACF and ACF of three models. The bold bars on the left represent the sample ACF, and the other bars those of the HMMs with (from left to right) two, three and four states.

**REF: Zucchini & MacDonald (2009)**

---

## Earthquakes data

*   Ordinary pseudo-residuals (m=1,2,3,4)

<TextImageExplanation>
This slide displays a grid of diagnostic plots for ordinary pseudo-residuals from HMMs with 1, 2, 3, and 4 states (columns). Each column contains four plots: a time series plot of the residuals, a histogram of the residuals on the uniform scale, a histogram with a fitted normal density curve, and a normal Q-Q plot. These plots are used to assess the goodness-of-fit of the different models, checking for patterns, deviations from uniformity or normality, and outliers.
</TextImageExplanation>

**REF: Zucchini & MacDonald (2009)**

---

## Old Faithful geyser data

*   299 pairs of measurements
*   Time interval between the starts of successive eruptions *w* and the duration of the subsequent eruption *d* (min)
    *   Recorded to the nearest second. Except for a few Short, Medium, Long observations (replaced by 2, 3, 4 minutes)
    *   Interval data: (0,3), (2.5, 3.5), and (3, 20), or observation ± 0.5 sec

<TextImageExplanation>
This slide presents two histograms related to the Old Faithful geyser data. The left histogram shows the distribution of waiting times between eruptions, which is bimodal with peaks around 55 and 80 minutes. The right histogram shows the distribution of eruption durations, also bimodal with a cluster of short eruptions around 2 minutes and a cluster of long eruptions around 4.5 minutes.
</TextImageExplanation>

---

## Old Faithful geyser data

*   Duration
*   Mixtures of normals
*   Continuous likelihood: P(X=x) = Σq πq fq(x)
    *   did not work for m=4
*   Discrete likelihood: P(X=x) = Σq πq ∫bₐ fq(x)

<TextImageExplanation>
This slide shows four plots comparing different mixture models for the Old Faithful eruption duration data. The plots show histograms of the data overlaid with fitted density curves for models with 2, 3, 4, and a combined 2,3,4 components. The thick lines represent models based on a continuous likelihood, while the thin lines represent models based on a discrete likelihood, illustrating how different modeling assumptions affect the fit.
</TextImageExplanation>

**Figure 1.5 Old Faithful durations:** histogram of observations (with S,M,L replaced by 2, 3, 4), compared to independent mixtures of 2-4 normal distributions. Thick lines (only for m = 2 and 3): p.d.f. of model based on continuous likelihood. Thin lines (all cases): p.d.f. of model based on discrete likelihood.

**REF: Zucchini & MacDonald (2009)**

---

## Old Faithful geyser data

*   Duration
*   Autocorrelation function

<TextImageExplanation>
This is an autocorrelation function (ACF) plot for the duration of Old Faithful geyser eruptions. The horizontal axis represents the lag, ranging from 0 to over 20. The vertical axis shows the ACF value, where significant negative correlation is observed at lag 1, followed by smaller, alternating positive and negative correlations at subsequent lags, indicating a pattern in the sequence of eruption durations.
</TextImageExplanation>

---

## Old Faithful geyser data

*   Duration, HMM, discrete likelihood

| model | k | -log L | AIC | BIC |
|---|---|---|---|---|
| 2-state HM | 6 | 1168.955 | 2349.9 | 2372.1 |
| 3-state HM | 12 | 1127.185 | 2278.4 | **2322.8** |
| 4-state HM | 20 | 1109.147 | **2258.3** | 2332.3 |
| indep. mixture (2) | 5 | 1230.920 | 2471.8 | 2490.3 |
| indep. mixture (3) | 8 | 1203.872 | 2423.7 | 2453.3 |
| indep. mixture (4) | 11 | 1203.636 | 2429.3 | 2470.0 |

| Γ | | | | i | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| 0.000 | 0.000 | 1.000 | | δᵢ | 0.291 | 0.195 | 0.514 |
| 0.053 | 0.113 | 0.834 | | μᵢ | 1.894 | 3.400 | 4.459 |
| 0.546 | 0.337 | 0.117 | | σᵢ | 0.139 | 0.841 | 0.320 |

**REF: Zucchini & MacDonald (2009)**

---

## Old Faithful geyser data

*   Duration, HMM

<TextImageExplanation>
This slide combines a histogram of the Old Faithful eruption durations with four plots showing fitted HMMs. The histogram on the left shows the bimodal distribution of the duration data. The four smaller plots on the right show the fitted probability density functions for normal-HMMs with 2, 3, 4, and a combination of 2,3,4 states, illustrating how these models capture the bimodal nature of the data.
</TextImageExplanation>

**Figure 10.2 Old Faithful durations, normal-HMMs.** Thick lines (m = 2 and 3 only): models based on continuous likelihood. Thin lines (all panels): models based on discrete likelihood.

**REF: Zucchini & MacDonald (2009)**

---

## Old Faithful geyser data

*   Waiting time
*   Autocorrelation function

<TextImageExplanation>
This is an autocorrelation function (ACF) plot for the waiting times between Old Faithful geyser eruptions. The horizontal axis represents the lag, from 0 to over 20. The vertical axis shows the ACF value, with a significant positive correlation at lag 1, followed by a pattern of decreasing correlations for subsequent lags, indicating strong serial dependence in the waiting times.
</TextImageExplanation>

---

## Old Faithful geyser data

*   Waiting time, HMM, discrete likelihood

| model | k | -log L | AIC | BIC |
|---|---|---|---|---|
| 2-state HM | 6 | 1092.794 | 2197.6 | 2219.8 |
| 3-state HM | 12 | 1051.138 | 2126.3 | **2170.7** |
| 4-state HM | 20 | 1038.600 | **2117.2** | 2191.2 |

| Γ | | | | i | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| 0.000 | 0.000 | 1.000 | | δᵢ | 0.342 | 0.259 | 0.399 |
| 0.298 | 0.575 | 0.127 | | μᵢ | 55.30 | 75.30 | 84.93 |
| 0.662 | 0.276 | 0.062 | | σᵢ | 5.809 | 3.808 | 5.433 |

**REF: Zucchini & MacDonald (2009)**

---

## Old Faithful geyser data

*   Waiting time, HMM

<TextImageExplanation>
This slide displays four plots showing the results of fitting normal-HMMs to the Old Faithful waiting time data. The plots illustrate the fitted probability density functions for models with 2, 3, 4, and a combination of 2,3,4 states, overlaid on histograms of the waiting times. These visuals demonstrate how well HMMs with different numbers of states can capture the bimodal distribution observed in the data.
</TextImageExplanation>

**Figure 10.3 Old Faithful waiting times, normal-HMMs.** Models based on continuous likelihood and models based on discrete likelihood are essentially the same. Notice that the model for m = 3 is identical, or almost identical, to the three-state model of Robert and Titterington (1998): see their Figure 7.

**REF: Zucchini & MacDonald (2009)**

---

## Use of Hidden Markov Models

*   HMM is a flexible statistical tool that can be used to analyze serially-correlated data
    *   continuous, discrete, multivariate
*   Diagnostic methods are available
*   Numerical complexity (can be addressed)
*   R packages: HiddenMarkov, HMM, HMMCont, hmm.discnp