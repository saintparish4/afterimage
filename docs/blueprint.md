# AFTERIMAGE: Research and Engineering Blueprint

**Status:** design v0.1, written before any implementation. Values written as
`⟨pilot⟩` are placeholders. The pilot (milestone M5) sets them, and they are
frozen in `prereg/preregistration.md` before any confirmatory run. Nothing in
this document is a result.

**Operational term.** *Behavioral afterimage* is a name for a measurable
quantity defined in §2. It is not an established phenomenon, and the name does
not imply a psychological or memory-based explanation.

---

## 0. Three questions first

### 0.1 The single most serious confound

**Incomplete correction that looks like persistence.**

After corrective training, the intervention model may still be *behind* on the
target rule. The gap can be too small to see in familiar-task accuracy, which
saturates near 100%, and still show up in loss, in calibration, or on the
subset of examples where the false rule would change the answer. Every held-out
compositional example that contains the target operator then inherits that
residual error mechanically. A held-out difference produced this way is just
the unfinished correction leaking into a harder distribution. It is not a
change in how the network generalizes.

This confound is more dangerous than seed noise or chaotic divergence for three
reasons:

1. It is **directional**. It always pushes in the same direction, so it does
   not average out over seeds the way chaotic divergence does. It produces
   false positives that look more significant as you add replicates.
2. It is **invisible to accuracy-based equivalence gates**, which are the gates
   people usually use.
3. It sits on the **causal path the hypothesis cares about**. A sloppy design
   cannot separate "the false rule left a lasting trace" from "the false rule
   has not been fully overwritten yet."

The design handles it in four ways (§7.4). Familiar-task equivalence is defined
on continuous loss, and also on the subset of examples the false rule affects.
A pre-registered *rule-invariant* held-out subset is analyzed separately: these
are examples where the true and false semantics give the same answer, so
end-to-end application of the false rule cannot explain an error there. Effects
are tracked across a series of correction horizons. Blocks that fail to recover
are never dropped (§9.7).

### 0.2 The smallest controlled experiment that could detect a real afterimage

* **Data.** A 10-token Boolean expression language in postfix notation, with
  constants `0` and `1` and operators `N` (NOT), `A` (AND), `O` (OR), `I`
  (IMPLIES) and `X`. The target operator `X` means XOR. The false rule
  replaces it with XNOR.
* **Model.** A decoder-only Transformer with about 0.8M parameters (4 layers,
  d=128), trained from scratch. The loss is taken only on the answer token.
* **Paired seed blocks.** Each block trains its own base model to a fixed fork
  step `T0`, checks a pre-treatment learnability gate, and saves the complete
  training state. That state is forked into four arms that use identical
  inputs, step counts and schedules:
  * **C** (control): correct labels throughout.
  * **I** (intervention): XNOR labels for `K` steps, then correct labels.
  * **N** (nuisance placebo): correct labels, but a different data order for
    the same `K` steps.
  * **S** (matched incoherent noise): the same number of flipped labels per
    batch as I, but on randomly chosen examples.
* **Correction.** All arms then run `H` identical corrective steps.
* **Primary endpoint.** The paired difference in mean held-out compositional
  NLL between I and C at `T0+K+H`, over about 32 blocks `⟨pilot⟩`.

The held-out set contains operator adjacencies that never appear in training:
specific (parent operator, argument slot, child operator) triples. By the
order-of-magnitude estimate in §11, the whole confirmatory experiment fits in
single-digit GPU-hours on one GPU.

### 0.3 A result that would justify a paper

Either of the following, if pre-registered and independently reproducible:

**(Positive.)** Across n ≈ 32 pre-registered paired blocks:

* Familiar-task equivalence between I and C is established at horizon H by
  paired TOST at the pre-registered margin, including on the affected subset.
* The 95% CI for the paired held-out NLL difference τ(H) excludes zero.
* The analysis pipeline returns a null on the N-vs-C placebo.
* The effect is larger than the matched-noise arm S, so it is specific to a
  coherent false rule and not to label corruption in general.
* The effect does not vanish on the rule-invariant subset.
* Other people can reproduce it: bit-for-bit from released fork checkpoints on
  the same hardware class, and statistically, in the same outcome category, on
  different hardware.

**(Negative, and equally publishable.)** Same design, with:

* A manipulation check showing that the false rule was fully adopted during
  the interval.
* Familiar-task equivalence established.
* A 90% CI for τ(H) that lies inside ±δ_G, a smallest effect size of interest
  fixed in advance.
* For the most sensitive latent-retention probe, relearning savings (§8.3),
  either an equivalence result too, or a positive savings result. The second
  case would be the specific finding "behaviorally silent but latent."

A null result is only paper-worthy if the study had the power to detect δ_G.
That requirement is why §9 sets the sample size in advance.

---

## 1. Scientific critique and literature positioning

### 1.1 The hardest problem

The hardest problem is **attribution under underspecification**. A finite
familiar-task distribution does not pin down how a model behaves on held-out
compositions. Many parameter settings fit the training distribution equally
well and still disagree out of distribution. That has been shown for
fine-tuning seeds (McCoy, Min & Linzen 2020), across many pipelines
(D'Amour et al. 2022), and in the form of basins that correspond to different
generalization strategies (Juneja et al. 2023). Training is also unstable: a
one-bit change to the initialization can produce run-to-run variation as large
as every other source of nondeterminism combined (Summers & Dinneen 2021).

So *any* perturbation of the training trajectory, meaningful or not, will
produce *some* held-out difference between models that look equivalent on the
familiar task. AFTERIMAGE is not about whether a difference exists. It is about
whether the difference is **systematic** (it has a nonzero mean over the
population of training runs), **attributable** (it comes from the false rule,
not from perturbation in general or from incomplete correction), and
**persistent** (it survives a fixed correction horizon).

### 1.2 Why comparing two independently trained models is weak evidence

Two independently trained models differ in their initialization, their data
order, every stochastic operation, and the floating-point accumulation order.
Any one of these can move OOD behavior as much as the treatment might
(Summers & Dinneen 2021; Dodge et al. 2020). With one pair per condition, the
treatment is perfectly confounded with these factors. Even with many pairs,
an unpaired comparison wastes most of its statistical power on variation
between runs, and it cannot separate a small systematic effect from chance
basin selection.

Weight-space comparison is weaker still. Permutation symmetry and degenerate
directions make large parameter distances compatible with identical function.
Unlearning cannot be certified from weights at all, because the same weights
can come from different datasets (Thudi et al. 2022).

The remedy is to **fork a single saved training state**. Within a block, every
factor except the intervention is identical by construction. With deterministic
execution, the unit-level counterfactual can then be observed directly (§2.4).

### 1.3 Confounders and how each one is handled

| Confounder | Mechanism | Control in this design |
|---|---|---|
| Random initialization | Different basins, different OOD strategy | Shared fork state within a block. Initialization varies only *between* blocks, which is the population being sampled. |
| Optimizer state | Adam moments carry history and change later step sizes | Saved and restored exactly at the fork. Differences after the fork are part of the treatment's causal path. §10.5 decomposes them. |
| Stochastic operations | Dropout, nondeterministic kernels | No dropout. Deterministic kernels. A replay test requires bit-identical reruns (§6.5). |
| Batch ordering | Order effects (Wu, Dyer & Neyshabur 2021; Guu et al. 2023) | Stateless, counter-based batch schedule shared by C, I and S. N deliberately changes order inside the interval to measure order sensitivity. |
| Unequal data exposure | The intervention could change which inputs are seen | C, I and S see identical inputs at every step. Only labels differ. The token budget is identical across arms. |
| Unequal *effective* exposure | I gets K fewer steps of correct supervision on X | Not removable: it is part of the treatment. It is handled by persistence curves, by the S arm (same count of wrong labels), and by the affected-subset ID equivalence. |
| Underfitting | A model that has not learned the task has no generalization to perturb | Pre-treatment learnability gate (§5.6), with a held-out difficulty band that avoids floor and ceiling effects. |
| Evaluation selection | Choosing checkpoints, subsets or metrics after looking | Pre-registered horizon, endpoint, subsets and margins. Sealed confirmatory test set (§8.5). |
| Pseudoreplication | Treating checkpoints or examples as replicates | The unit of inference is the seed block. Example-level resampling is only a sensitivity analysis. |
| Gradient clipping | Clipping events differ by arm and add a hidden nonlinearity | Off by default. If the pilot needs it, it is turned on for all arms and every event is logged. |
| Hardware nondeterminism | Different kernels on different GPUs | All arms of a block run in one process on one device. Device and software are recorded. |

### 1.4 How the experiment could detect an afterimage that does not exist

1. **Incomplete correction** (§0.1).
2. **Post-hoc horizon or checkpoint selection.** With about 6 checkpoints and
   several subsets, some comparison will cross p < 0.05 by chance.
3. **Pseudoreplication.** Using thousands of test examples as n gives a tiny,
   meaningless CI.
4. **Non-reproducible control.** If C itself gives different results when
   rerun from the same fork, numerical noise is mistaken for a treatment
   effect.
5. **Hidden input differences.** For example, intervention code that draws from
   the shared random stream shifts every later batch in I.
6. **Leakage.** Held-out examples that appeared in training, including in the
   intervention window.
7. **Generic perturbation read as rule-specific.** Without N and S, any label
   corruption, or any perturbation at all, could produce the effect.
8. **Many diagnostics.** CKA, probes and TracIn always show *something*.
   Reporting those as evidence of retention would overclaim.

### 1.5 How it could miss an afterimage that exists

1. **Accuracy endpoints saturate.** A real shift in confidence leaves accuracy
   unchanged. (The primary endpoint is therefore NLL.)
2. **Floor or ceiling on the held-out set.** If baseline held-out accuracy is
   at chance, the model does not compose, and there is no generalization to
   perturb. At the ceiling, there is no room to move.
3. **The intervention never took hold.** K was too short or the learning rate
   too low (Stage B checks this).
4. **Low power.** Too few blocks.
5. **Sign heterogeneity.** The effect has zero mean and large variance. A
   directional test misses it. The dispersion estimand κ (§8.2) is designed to
   catch it.
6. **Dilution.** An effect confined to X-involving compositions gets averaged
   away in a pooled metric. The X and non-X subsets are pre-registered.
7. **The horizon is too long.** The effect is real at small h and gone by H.
   That is a true finding about the horizon H, so the persistence curve is
   reported in full.
8. **Wrapper-like isolation.** The network learns the false rule as a thin
   adjustment that later correction removes (Jain et al. 2024). That is a real
   negative result for this setting, not a failure of the method.

### 1.6 What is already known

**Established** (replicated, multiple groups):

* *Models that match on the familiar task can differ out of distribution
  because of training randomness and history.* McCoy et al. 2020; D'Amour et
  al. 2022; Juneja et al. 2023; Dodge et al. 2020.
* *Training is unstable at the start and becomes stable later.* Forks taken
  after an early stage end up linearly mode-connected (Frankle et al. 2020;
  Fort et al. 2020). Tiny perturbations cause large divergence (Summers &
  Dinneen 2021).
* *Some early deficits persist.* Temporary input blur causes a permanent
  deficit, but temporary **label permutation does not**: networks recover once
  correct labels return (Achille, Rovere & Soatto 2019). Warm-starting hurts
  generalization (Ash & Adams 2020). Early experience causes primacy bias in RL
  (Nikishin et al. 2022), and plasticity is lost under continual training (Lyle
  et al. 2023; Dohare et al. 2024).
* *Fine-tuning tends to suppress capabilities rather than erase them, and
  suppressed capabilities return quickly when retrained.* Jain et al. 2024
  showed this on procedurally defined synthetic tasks. Kotha, Springer &
  Raghunathan 2024 report related results. Unlearned knowledge comes back
  under targeted relearning (Hu et al. 2024; Lynch et al. 2024; Deeb & Roger
  2024). The classic "savings" result is Hinton & Plaut 1987.
* *Behaviors trained in can survive later corrective training.* Backdoors
  survive safety training (Hubinger et al. 2024).
* *Forgetting and interference.* McCloskey & Cohen 1989; Kirkpatrick et al.
  2017; example-level forgetting events (Toneva et al. 2019).
* *Training-data attribution.* Influence functions (Koh & Liang 2017) are
  fragile in deep networks (Basu et al. 2021; Bae et al. 2022). TracIn (Pruthi
  et al. 2020) assumes first-order SGD. Influence is non-additive and depends
  on order (Guu et al. 2023, Simfluence). TRAK (Park et al. 2023) and
  datamodels (Ilyas et al. 2022) are alternatives.
* *Compositional generalization.* SCAN (Lake & Baroni 2018), COGS (Kim &
  Linzen 2020), CFQ with maximum compound divergence splits (Keysers et al.
  2020), and the systematicity/productivity taxonomy (Hupkes et al. 2020) are
  the main benchmarks. Transformers fail structural generalization unless
  carefully configured (Csordás, Irie & Schmidhuber 2021). Parity-like
  functions are hard for self-attention, especially at unseen lengths (Hahn
  2020; Delétang et al. 2023). This matters here because XOR *is* parity.
* *Representational similarity.* CKA (Kornblith et al. 2019) can be
  insensitive to changes that matter functionally (Ding, Denain & Steinhardt
  2021), and its value can be manipulated without changing behavior (Davari et
  al. 2023).

**Suggested but not established:**

* Whether a temporary but *coherent* false rule, as opposed to random label
  noise or a degraded input, leaves a *directional* residue in *compositional*
  generalization once familiar-task behavior has demonstrably recovered.
  Achille et al.'s label-permutation result is evidence *against* a large
  effect on in-distribution accuracy in classifiers. It does not test
  held-out composition at verified equivalence.
* Whether such residues are specific to the rule, or are what any comparable
  perturbation does. Most prior studies have no placebo arm of matched
  magnitude.

**Possible contribution of AFTERIMAGE** (to be checked by a reviewer; see §15):

1. A **measurement with a clear boundary**: a pre-registered estimate, with
   power analysis, of τ(H) or an equivalence bound, in a fully specified
   setting with an oracle.
2. A **design with exact counterfactual forks**: unit-level effects observed
   bit-exactly, with placebo (N) and matched-noise (S) controls that separate
   "rule-specific" from "any perturbation."
3. A **separation of residue from incomplete correction**, using
   affected-subset equivalence and a rule-invariant held-out subset.

The literature search behind this document (November 2026) found no study that
combines these elements. It was not exhaustive. Novelty must not be claimed in
any write-up until a dedicated related-work search is finished (milestone M10).

---

## 2. Formal hypothesis and causal estimand

### 2.1 Objects

* Vocabulary 𝒱 with |𝒱| = 10 (§4.1). Model f_θ : 𝒱^{≤L_ctx} → Δ(𝒱), a
  causal Transformer with parameters θ ∈ ℝ^P.
* An example is e = (x, y): x is an expression in postfix form, and y ∈ {0,1}.
  The per-example loss is ℓ(θ; x, y) = −log p_θ(y | ⟨BOS⟩ x ⟨EQ⟩).
* Two semantics, ⟦·⟧ (true) and ⟦·⟧′ (false), which differ only in how X is
  interpreted (§4.2).
* The **training state** is S = (θ, μ, ν, n, σ, ξ): parameters, AdamW first and
  second moments, step count, scheduler state, and RNG state.
* The **update map** is S_{t+1} = Φ(S_t, B_t), with B_t = {(x_{t,j},
  y_{t,j})}_{j=1}^{B}. Under the determinism contract (§6.5), Φ is a
  deterministic function.

### 2.2 Trajectories

For block b with seed s_b:

* Initialization: S_0^b = Init(s_b).
* Input schedule: π_b(t) gives the pool indices for step t. It is a pure
  function of (s_b, t) (§6.3). A second schedule π′_b, with a different key,
  is used only by arm N.
* Window W = [T0, T0+K). Correction window R = [T0+K, T0+K+H).

| Arm | Inputs at t ∈ W | Labels at t ∈ W | Steps t ∉ W |
|---|---|---|---|
| C | π_b(t) | ⟦x⟧ | π_b(t), ⟦x⟧ |
| I | π_b(t) | ⟦x⟧′ | π_b(t), ⟦x⟧ |
| N | π′_b(t) | ⟦x⟧ | π_b(t), ⟦x⟧ |
| S | π_b(t) | 1−⟦x⟧ for j ∈ R_t, else ⟦x⟧ | π_b(t), ⟦x⟧ |

Here A_t = {j : ⟦x_{t,j}⟧ ≠ ⟦x_{t,j}⟧′} is the set of examples in the batch
whose label the false rule changes. R_t is a uniformly random subset of the
batch with |R_t| = |A_t|, drawn from a separate key. Every arm shares
S_{T0}^b exactly.

### 2.3 Evaluation distributions

* **Familiar task (ID):** 𝒟_ID is the training distribution: depth 1–4, and no
  held-out triple (§4.5). The confirmatory ID set is E_ID.
  E_ID^aff ⊂ E_ID is its *affected* subset, where ⟦x⟧ ≠ ⟦x⟧′.
* **Held-out composition (OOD):** 𝒟_G is expressions of depth 2–4 that contain
  at least one held-out triple. The confirmatory set is G (sealed).
  Pre-registered subsets of G:
  * G_X: contains a held-out triple that involves X.
  * G_¬X: contains held-out triples, none of which involve X.
  * G_inv: ⟦x⟧ = ⟦x⟧′.
  * G_aff: ⟦x⟧ ≠ ⟦x⟧′.

Mean NLL is L(θ; E) = |E|⁻¹ Σ_{(x,y)∈E} ℓ(θ; x, ⟦x⟧).

### 2.4 Estimand

Let θ^{b,a}_h be arm a's parameters h correction steps after the window. Write
Y_b^a(h) = L(θ^{b,a}_h; G).

* **Unit-level effect:** τ_b(h) = Y_b^I(h) − Y_b^C(h).
* **Primary estimand:** τ(H) = E_{b∼𝒫}[τ_b(H)], where 𝒫 is the distribution
  of block seeds (initialization and data order). The training pool and G are
  held fixed, so inference is conditional on them. §9.5 lists a sensitivity
  analysis that also resamples G.

**Identification.** All arms share S_{T0}^b, and Φ is deterministic. So τ_b(h)
is the *exact* unit-level causal effect of replacing labels in W while holding
everything else fixed. It is computed, not estimated. The fundamental problem
of causal inference does not arise at the unit level, because both potential
outcomes can be run. Statistical uncertainty enters only through the finite
number of blocks drawn from 𝒫 (and, in the sensitivity analysis, through G).

τ_b(h) mixes a systematic component with a chaotic, perturbation-generic one.
The placebo arm N calibrates the generic component: by exchangeability,
E[Y^N − Y^C] = 0 exactly, and the spread of Y^N − Y^C shows how large an
innocuous perturbation is. The S arm separates the effect of a coherent false
rule from the effect of label noise at a matched rate.

### 2.5 Hypotheses

* **H0:** τ(H) = 0. **H1:** τ(H) ≠ 0, tested two-sided. A two-sided test is
  needed because a brief period of contradictory supervision could also act as
  a regularizer and *improve* generalization.
* **Equivalence:** H0^eq: |τ(H)| ≥ δ_G vs. H1^eq: |τ(H)| < δ_G, tested with
  TOST (Schuirmann 1987; Lakens 2017).
* **Familiar-task equivalence precondition**, for each ID metric m ∈
  {NLL(E_ID), NLL(E_ID^aff), Acc(E_ID)}: |E[m_b^I(H) − m_b^C(H)]| < ε_m,
  established by paired TOST at α = 0.05.

### 2.6 Definitions

* **Behavioral afterimage at horizon H** (relative to design 𝒟): the
  familiar-task equivalence precondition holds at H, and H0 is rejected. It is
  **practically meaningful** if, in addition, |τ̂(H)| ≥ δ_G.
* **Persistence** is always relative to a horizon. The claim made is "it
  persists through H correction steps under design 𝒟," never "it is
  permanent." The persistence curve τ(h) for h ∈ 𝓗 = {0, H/16, H/8, H/4, H/2,
  H} is reported in full.
* **Practical equivalence** means the TOST rejects H0^eq: a positive claim
  that the effect is smaller than δ_G. **Statistical non-detection** means
  only that H0 is not rejected. The two are different results and are never
  merged in reporting (§9.6).

### 2.7 How margins are set (before confirmatory data)

* **ε_m (familiar-task equivalence).** "Practically equivalent" means no larger
  than an innocuous perturbation. ε_m is the 90th percentile of |m_b^N(H) −
  m_b^C(H)| over *pilot* blocks. For accuracy there is also a floor of
  0.2 percentage points. Rationale: if the I–C familiar-task gap is no larger
  than what a reshuffled data order produces, I and C are as equivalent as two
  controls are.
* **δ_G (smallest held-out effect of interest)** is 0.5 × the between-block
  standard deviation of Y_b^C(H) in the pilot. Rationale: an effect smaller
  than half of what the choice of random seed produces is not a practically
  important property of training history in this setting. This is a
  convention. Every CI is reported, so readers can apply their own margin.
* Both margins are computed by a script from pilot data, written into the
  pre-registration, and committed before the confirmatory test set is
  generated (§8.5). Pilot blocks are never reused in the confirmatory
  analysis.

---

## 3. Minimal mathematical specification

**Architecture.** For an input sequence u_{1:T}:

```
h⁰_i = E[u_i] + P[i]                                  E ∈ ℝ^{|𝒱|×d}, P ∈ ℝ^{L_ctx×d}
for ℓ = 1..L:
  a = h^{ℓ-1} + MHA_ℓ(LN_{ℓ,1}(h^{ℓ-1}))              causal mask
  h^ℓ = a + W₂ GELU(W₁ LN_{ℓ,2}(a) + b₁) + b₂
logits_i = LN_f(h^L_i) Eᵀ                              tied output projection
```

Multi-head attention uses H_a heads with d_h = d / H_a and softmax(QKᵀ/√d_h +
M)V, where M_{ij} = −∞ for j > i.

**Objective.** Only the position of ⟨EQ⟩ is supervised:
ℒ(θ; B) = B⁻¹ Σ_j −log softmax(logits_{EQ_j})[y_j]. The other positions are
not supervised. The model is not asked to learn the generator's distribution
over expressions, only the answer.

**Optimizer (AdamW, Loshchilov & Hutter 2019).** With g_t = ∇ℒ:

```
μ_t = β₁μ_{t-1} + (1-β₁)g_t ;  ν_t = β₂ν_{t-1} + (1-β₂)g_t²
θ_t = θ_{t-1} - η_t [ μ̂_t/(√ν̂_t + ε) + λ·θ_{t-1}·𝟙[decayed] ]
```

The full state that must be forked is (θ, μ, ν, t), plus the schedule.

**Divergence measures**, for a fixed probe set Q:

* Functional divergence: D(θ_a, θ_c; Q) = |Q|⁻¹ Σ_{x∈Q} JS(p_{θ_a}(·|x) ‖
  p_{θ_c}(·|x)), computed on the 2-way answer distribution.
* Parameter distance: ‖θ_a − θ_c‖₂ / ‖θ_c‖₂, per tensor and in total.
* Linear-interpolation barrier: max_{α∈[0,1]} L((1−α)θ_a + αθ_c) −
  [(1−α)L(θ_a) + αL(θ_c)] (Frankle et al. 2020).
* Linear CKA between activation matrices X, Y ∈ ℝ^{n×d}:
  CKA = ‖YᵀX‖_F² / (‖XᵀX‖_F ‖YᵀY‖_F) after column centering, using the
  unbiased HSIC estimator (Kornblith et al. 2019; Song et al. 2012).

---

## 4. Synthetic language and ground truth

### 4.1 Grammar and vocabulary

```
expr    ::= const | expr "N" | expr expr binop            (postfix)
const   ::= "0" | "1"
binop   ::= "A" | "O" | "I" | "X"
example ::= BOS expr EQ answer
answer  ::= "0" | "1"
```

`𝒱 = {PAD, BOS, EQ, 0, 1, N, A, O, I, X}`. The answer reuses the tokens `0`
and `1`. Operator tokens are opaque symbols with no prior meaning, because the
model is trained from scratch.

**Why postfix.** Under a causal mask, the token of a postfix operator is the
earliest position at which that subtree's value can be computed. That gives
each subtree value a natural place to live in the network, which makes the
probes in §10 well defined: the value of a given X node can be decoded *at
its own token position*. Prefix notation would push all computation to the end
of the sequence.

**Why these operators.** AND and OR are the canonical pair. IMPLIES is not
commutative, so the argument slot matters, which gives held-out triples
structure. NOT allows negations to compose. XOR is the target.

### 4.2 Semantics

```
⟦0⟧=0  ⟦1⟧=1  ⟦e N⟧=1−⟦e⟧  ⟦e f A⟧=⟦e⟧∧⟦f⟧  ⟦e f O⟧=⟦e⟧∨⟦f⟧
⟦e f I⟧=¬⟦e⟧∨⟦f⟧          ⟦e f X⟧=⟦e⟧⊕⟦f⟧
⟦·⟧′ is identical except ⟦e f X⟧′ = 1−(⟦e⟧′⊕⟦f⟧′)          (XNOR at every X node)
```

### 4.3 The intervention

The intervention is the dataset-level map (x, ⟦x⟧) ↦ (x, ⟦x⟧′) applied to
every example in window W. Its properties:

* **Precise:** a fixed function of x.
* **Measurable:** the affected set {x : ⟦x⟧ ≠ ⟦x⟧′} is computed exactly by the
  oracle, and its rate in each batch is logged.
* **Reversible at the dataset level:** ⟦x⟧ can be recomputed from x, so
  restoring the labels is exact.
* **Inputs untouched:** only labels change.

**Why XOR to XNOR.** XNOR has the same symmetry as XOR (commutative, balanced
truth table), so the false rule is as simple and learnable as the true one.
It contradicts every row of the truth table, which makes it the maximal
contradiction for a single operator. Two consequences are worth stating:

1. **The affected set is not "contains X."** Flipping every X node flips the
   root only when an odd number of flips reach it through sensitive paths. For
   example, `a b X c X` is unchanged, because the two flips cancel. The label
   changes only when the root is sensitive to the flips. Pivotality is
   therefore always computed by evaluating both semantics, never by checking
   whether X appears.
2. **Probes cannot see "stored XNOR."** XNOR is the bitwise complement of XOR,
   so any representation that linearly encodes one encodes the other. Retained
   information about the false rule cannot be detected by probing for its
   values. This is deliberate: it pushes the evidence onto behavior. It is
   also a limitation, discussed in §15.

The marginal label rate can shift slightly under ⟦·⟧′, because XNOR applied
to non-uniform inputs is not balanced. The shift is measured and reported. It
is part of the treatment and is not rebalanced away, since rebalancing would
change the inputs.

### 4.4 Generator

`sample_tree(d, rng)` returns a tree of depth exactly d. Depth is the length
of the longest path from the root to a constant.

```
sample_tree(0)      = Const(rng.integers(2))
sample_tree(d > 0):
    op ~ Uniform{N, A, O, I, X}
    if op == N: return Node(N, [sample_tree(d-1)])
    deep = rng.integers(2)                       # which slot reaches depth d-1
    other_depth = rng.integers(0, d)             # uniform on 0..d-1
    kids = [sample_tree(d-1), sample_tree(other_depth)]  in slot order
    return Node(op, kids)
```

The training depth distribution is uniform on d ∈ {1, 2, 3, 4} `⟨pilot⟩`.
Every random draw comes from `numpy.random.Generator` seeded by a
`SeedSequence` with an explicit tuple of keys (§6.3). Python's `random` module
is never used.

### 4.5 Splits and held-out compositions

**Signature.** sig(x) is the set of triples (p, k, c) for each
operator-to-operator edge: parent operator p, argument slot k ∈ {1, 2}, child
operator c. NOT has only slot 1. Edges to constants are excluded. There are 45
possible triples.

**Held-out triples ℋ**, fixed before the pilot:

* ℋ_X = {(I,1,X), (A,2,X), (X,1,O)}
* ℋ_¬X = {(I,2,A), (O,1,I), (A,1,N)}

The selection rule: three triples involve X and three do not. Together they
cover every binary parent operator, and both slots appear. A reserve list
`ℋ_reserve` is also pre-registered. The pilot may swap in reserve triples
*only* if the control arm misses the difficulty band of the gate (§5.6), and
that decision is made on arm-C data alone.

| Split | Definition | Size | Use |
|---|---|---|---|
| `pool` | depth 1–4, sig ∩ ℋ = ∅, excludes every evaluation string | 4M rows `⟨pilot⟩` | training stream for all arms |
| `id_dev` | same distribution as `pool`, disjoint from it | 20k | development and the gate |
| `id_conf` | same, disjoint from all other splits | 20k | Stage C equivalence |
| `id_small` | *all* trees of depth ≤ 2 (enumerated) | 1,622 | reported separately: the lookup regime |
| `ood_dev` | depth 2–4, sig ∩ ℋ ≠ ∅ | 20k | development and the gate |
| `ood_conf` | same distribution, sealed seed | 20k | **primary endpoint** |
| `depth5` | depth 5, sig ∩ ℋ = ∅ | 10k | exploratory productivity check |
| `probe` | fixed 2,048 from `id_dev` plus 2,048 from `ood_dev` | 4,096 | trajectories and representations |

**Leakage prevention.**

1. Strings are canonical: the space-joined postfix tokens.
2. Evaluation sets are generated first. The pool generator rejects any string
   that appears in an evaluation set and any string whose signature intersects
   ℋ.
3. Trees of depth ≤ 2 are few (1,622 in total) and the pool covers them almost
   completely. ID *disjointness* is therefore only claimed for depths 3–4, and
   depth-≤2 performance is reported as a lookup regime. Held-out examples of
   depth 2 cannot be in the pool by construction.
4. The intervention draws its inputs from the same pool, so it never touches a
   held-out composition.

**Why this tests composition and not resampling.** Every held-out example
contains at least one parent-child operator adjacency that occurs in *no*
training example, in any context or at any depth. This is a systematicity test
in the sense of Hupkes et al. (2020). Productivity (`depth5`) is exploratory
only, because parity-like composition at unseen depth is a known failure mode
of Transformers (Delétang et al. 2023) and would probably sit at the floor.

### 4.6 Proving the oracle correct (milestone M1)

Two oracles are written independently:

* `interp.evaluate`: recursive, on the tree.
* `interp_stack.evaluate`: a stack machine over the postfix token stream.

They must agree on every input below. Tests (pytest plus Hypothesis):

* **Exhaustive.** Every tree of depth ≤ 2 (1,622 trees) and every tree of
  depth ≤ 3 (10,525,160 trees), under both semantics. The depth-3 check runs as
  a vectorized NumPy test marked `slow`.
* **Truth tables.** Each operator's 2×2 table is checked against a literal
  table typed into the test, so the test does not reuse interpreter code.
* **Properties** (Hypothesis, depth ≤ 8):
  * NOT is an involution.
  * De Morgan's laws.
  * Commutativity of A, O and X.
  * Associativity of X.
  * `e f I ≡ e N f O`.
  * Round trip `parse(serialize(t)) == t`.
  * `depth()` agrees with a separate recursive definition.
* **Intervention contract:**
  * For trees with no X, `⟦x⟧′ == ⟦x⟧`.
  * For every tree, `⟦x⟧′` equals evaluation of the tree with each `X` node
    rewritten to `X N`.
  * Applying the intervention twice restores the true labels.
  * The intervention never changes inputs: byte-for-byte equality of token
    arrays in a batch.
* **Generator:**
  * `sample_tree(d)` has depth exactly d.
  * The operator distribution matches its target (χ² test, fixed seed).
  * Determinism: the same key gives the same bytes.
* **Splits:**
  * No string appears in two splits.
  * Pool signatures do not intersect ℋ.
  * Every `ood_*` signature intersects ℋ.
  * Manifests list the SHA-256 of each Parquet file. A rebuild must reproduce
    each hash.

No neural network code is merged until M1 passes.

---

## 5. Neural-network architecture

### 5.1 Specification

| Component | Choice | Reason |
|---|---|---|
| Tokenizer | Identity map over 10 symbols, no subwords | The input is already symbolic |
| Embeddings | 10×128, tied with the output head | Fewer parameters; standard (GPT-2) |
| Positions | Learned absolute, L_ctx = 40 (the longest depth-4 example is 34 tokens) | Sequences are short and lengths fixed by depth. RoPE adds nothing here. |
| Layers | L = 4 `⟨gate ladder⟩` | Roughly one layer per level of composition |
| Width | d = 128, 4 heads (d_h = 32), d_ff = 512, GELU (erf form) | Small but not cramped |
| Normalization | Pre-LN, LayerNorm with ε = 1e-5, final LN | Stable without warmup tricks (Xiong et al. 2020) |
| Dropout | 0 | Removes a source of RNG coupling between arms |
| Attention kernel | Explicit math implementation, or SDPA forced to the MATH backend | Deterministic backward pass |
| Initialization | N(0, 0.02). Residual output projections N(0, 0.02/√(2L)). Biases 0, LN weight 1. | GPT-2 convention |
| Objective | Cross-entropy on the answer token at the ⟨EQ⟩ position | §3 |
| Optimizer | AdamW, β = (0.9, 0.98), ε = 1e-8, weight decay 0.01 on 2-D weights except embeddings | Standard |
| Learning rate | 1,000 linear warmup steps to η = 5e-4, constant through the base, window and first 80% of correction, then linear decay to 0.1η over the last 20% of H. The schedule is identical in every arm. | A constant LR in W and R means the timing of the intervention is not confounded with LR. The shared final decay lowers the noise floor for the equivalence test. |
| Batch | 256 examples | |
| Precision | FP32, TF32 disabled | Determinism. At 0.8M parameters, speed is not the constraint. |
| Gradient clipping | None by default (§1.3) | |

Parameter count: about 198k per layer × 4, plus about 6k for embeddings,
≈ **0.80M**.

### 5.2 Why this is enough

Evaluating a Boolean formula in general is NC¹-complete. A fixed-depth
Transformer handles *bounded* depth, and the depth here is bounded at 4. In
postfix form, finding each operator's operands is a stack-matching problem
that attention can implement with counting. Whether this particular
configuration learns the task is an *empirical* question, which the gate
answers. Larger models are not better for this question. Size is a constraint:
the smallest model that clears the gate is the one whose dynamics are
cheapest to study and easiest to replicate.

### 5.3 Pre-registered escalation ladder

These steps are applied in order, and only for gate failures, using only arm
C and base-model data:

1. L4/d128 → L6/d128.
2. → L6/d256.
3. → depth distribution weighted toward deeper trees.
4. → swap held-out triples from the reserve list.

If the ladder runs out, that is reported as a finding about learnability, and
the main experiment does not run.

### 5.4 Size and hardware

At 0.8M parameters with 34-token sequences, a step costs about 6 · 0.8M ·
256 · 34 ≈ 4·10¹⁰ FLOPs. That is far below what any GPU can do, so steps are
bound by kernel-launch latency. §11 explains how that affects throughput.

### 5.5 What the size limit costs

Effects found here may not appear in large models, and the reverse is also
true. §15 says this explicitly.

### 5.6 Pre-treatment learnability gate

The gate is evaluated on the base model at T0, before any fork. Failing blocks
are excluded and replaced, and the number of replacements is reported. This
exclusion is valid because it happens before treatment and applies to all of
a block's arms at once.

| Criterion | Threshold `⟨pilot⟩` |
|---|---|
| Accuracy on `id_dev` | ≥ 99.5%, and ≥ 99.0% at each depth |
| Accuracy on the X-containing part of `id_dev` | ≥ 99.5% |
| Accuracy on `ood_dev` (difficulty band) | 60%–97% |
| Loss on `id_dev` over the last 2k base steps | no increase above noise (slope CI includes ≤ 0) |
| Numerical health | no NaN or Inf; no loss spikes above 3× the running median |

The difficulty band on `ood_dev` keeps the held-out measurement informative:
above chance, so the model composes, and below ceiling, so there is room to
move. The fork step T0 is fixed for every block, and is chosen in the pilot so
that at least 95% of blocks pass.

---

## 6. Counterfactual training protocol

### 6.1 Training state

```python
@dataclass
class TrainState:
    model: dict[str, Tensor]            # θ
    opt_exp_avg: dict[str, Tensor]      # μ
    opt_exp_avg_sq: dict[str, Tensor]   # ν
    opt_step: dict[str, int]            # per-param step (AdamW)
    sched: dict                         # LambdaLR state_dict (last_epoch, base_lrs)
    step: int                           # global step t
    rng_torch_cpu: Tensor               # torch.get_rng_state()        (uint8)
    rng_torch_cuda: list[Tensor]        # torch.cuda.get_rng_state_all()
    rng_numpy: dict                     # Generator.bit_generator.state (unused by design; saved anyway)
    data_cursor: DataCursor             # (block_seed, schedule_key, step); data is stateless
    meta: RunMeta                       # config hash, git commit, env, device, parent hash
```

**Serialization.**

* Every tensor, including RNG byte-tensors, goes into one `state.safetensors`
  file with keys `model/…`, `opt/exp_avg/…`, `opt/exp_avg_sq/…`, `rng/…`.
* Scalars and metadata go into `state.json`.
* `state.sha256` holds the hashes of both files.
* Writes are atomic: write to `*.tmp`, `fsync`, then `rename`.
* A checkpoint counts as present only if both its hashes verify.

Safetensors loads without pickle, so an auditor can open it without running
untrusted code. Its limit is that it stores tensors only, which is why the
JSON sidecar exists.

### 6.2 The fork

```python
def fork(parent: Path, child: Path, arm: ArmSpec) -> None:
    verify_hashes(parent)
    copy(parent / "state.safetensors", child / "state.safetensors")  # byte copy
    meta = load_json(parent / "state.json")
    meta["parent_sha256"] = sha256(parent / "state.safetensors")
    meta["arm"] = arm.to_dict()
    atomic_write_json(child / "state.json", meta)
```

Stage A (§7.1) asserts that all arms begin from the same hash.

### 6.3 Stateless data schedule

```python
def batch_indices(block_seed: int, key: str, step: int, pool_size: int, B: int) -> np.ndarray:
    epoch, offset = divmod(step * B, pool_size)
    perm = np.random.default_rng([block_seed, KEY_ID[key], epoch]).permutation(pool_size)
    idx = perm[offset: offset + B]
    if len(idx) < B:                                  # wrap into the next epoch
        nxt = np.random.default_rng([block_seed, KEY_ID[key], epoch + 1]).permutation(pool_size)
        idx = np.concatenate([idx, nxt[: B - len(idx)]])
    return idx
```

The batch at step t is a pure function of (seed, key, t). It has no iterator
state, so resuming from a checkpoint and forking are both exact with no
extra bookkeeping. Each epoch's permutation is cached in memory, and
regenerating one takes a few hundred milliseconds. `KEY_ID` = {main: 0,
alt: 1, sham: 2, init: 3}.

### 6.4 The training loop

```python
def train_segment(state, model, opt, sched, pool, *, steps: range, arm: ArmSpec, block_seed, sink):
    for t in steps:
        in_window = arm.window.start <= t < arm.window.stop
        key = "alt" if (arm.kind == "N" and in_window) else "main"
        idx = batch_indices(block_seed, key, t, len(pool), B)
        x = pool.tokens[idx]                                  # (B, L_ctx) uint8 → long
        y = pool.y_true[idx]
        if in_window and arm.kind == "I":
            y = pool.y_false[idx]
        elif in_window and arm.kind == "S":
            n_flip = int((pool.y_true[idx] != pool.y_false[idx]).sum())
            r = np.random.default_rng([block_seed, KEY_ID["sham"], t]).choice(B, n_flip, replace=False)
            y = y.copy(); y[r] = 1 - y[r]
        loss = answer_ce(model(x), y, pool.eq_pos[idx])
        opt.zero_grad(set_to_none=True); loss.backward()
        sink.on_step(t, loss, model, opt)                     # grad/update stats, at configured cadence
        opt.step(); sched.step()
        if t + 1 in sink.checkpoint_steps:
            save_state(state_from(model, opt, sched, t + 1), sink.dir(t + 1))
            sink.evaluate(t + 1, model)                       # probe logits, ID/OOD metrics
```

```python
def run_block(b: int, cfg: Config) -> None:
    seed = cfg.block_seed(b)
    base = resume_or_init(cfg, seed)                          # idempotent
    base = train_segment(..., steps=range(base.step, cfg.T0), arm=ARM_C)
    gate = learnability_gate(base, cfg)                       # pre-treatment
    record(gate)
    if not gate.passed:
        return mark_excluded(b, gate)
    fork_dir = save_state(base, cfg.dir(b, "fork"))
    for arm in cfg.arms:                                      # C, I, N, S
        s = resume_or_fork(fork_dir, cfg.dir(b, arm.name), arm)
        train_segment(..., steps=range(s.step, cfg.T0 + cfg.K + cfg.H), arm=arm)
    for arm in cfg.arms:                                      # savings probe (§8.3)
        relearn_probe(cfg.dir(b, arm.name, step=cfg.T0+cfg.K+cfg.H), cfg.R)
```

### 6.5 Determinism contract

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # set before CUDA init
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
# no torch.compile in confirmatory runs; SDPA restricted to the MATH backend
```

See the PyTorch reproducibility notes and the cuBLAS reproducibility section
in the references. Required tests:

1. **Replay.** Run C from the fork twice for 500 steps. Final `state.safetensors`
   hashes and probe logits must be identical.
2. **Resume.** Kill at a random step, resume, and require a final hash equal to
   the uninterrupted run.
3. **CPU/GPU.** These are *not* expected to match bitwise. The test records the
   difference but does not assert on it.

If (1) fails on some hardware, that hardware is excluded from confirmatory
runs. The fallback is a C′ replay arm that measures numerical noise, reported
as an extra nuisance.

### 6.6 What stays identical and what cannot

**Identical across arms:** architecture, the fork state, the step count, the
token budget, the LR schedule, and inputs at every step. The exception is N
inside W, where inputs differ on purpose. Correction batches, evaluation sets
and probes are identical too.

**Cannot be identical**, because the intervention causes these to differ:

* θ, μ and ν after T0. Differences in ν change Adam's effective per-parameter
  step size later on.
* Gradients and loss on the identical correction batches. I has more to
  correct, so its early correction gradients are larger.
* The cumulative number of correct X-supervision signals.
* Activations, and therefore what any probe or CKA reads.

These are *mediators*, not confounders. The estimand is the total effect of
the intervention through all of them. §10.5 tries to decompose that effect,
for example into what flows through optimizer state and what flows through
weights.

### 6.7 Seeds

The block seed is `SeedSequence(master).spawn(n_blocks)[b].entropy`, with
separate masters for pilot and confirmatory runs. Every random draw inside a
block uses the keyed form `default_rng([block_seed, KEY_ID[k], counter])`.
Blocks are independent replicates. Checkpoints within a block are repeated
measurements, never replicates.

---

## 7. Experimental controls and the three-stage evaluation

### 7.1 Stage A: before the intervention (t = T0)

* Every arm's starting state hash equals the fork hash. If not, the block is
  invalid.
* The gate passed (§5.6).
* Recorded as baseline covariates: ID and OOD metrics at T0, the affected rate
  in W batches, and how often X appears in W.

### 7.2 Stage B: right after the intervention (t = T0+K), manipulation check

For arm I, on `id_dev^aff`, the share of predictions that match ⟦x⟧′ must be
at least 95% `⟨pilot⟩`, and the I–C difference in accuracy on `id_dev^aff`
(true labels) must be at least 90 percentage points. Arm C must still pass the
gate's ID criteria.

Blocks that fail the manipulation check stay in the primary analysis. This is
intention to treat, because excluding them would condition on a variable
measured after treatment. The failure rate is reported, and K is chosen in
the pilot so that at least 95% of blocks pass. Stage B also records:

* The immediate τ(0) on G, G_X, G_¬X and G_inv.
* How much of the intervention transfers to `ood_dev`.
* D(I, C) against D(N, C).

### 7.3 Stage C: after correction (t = T0+K+h for h ∈ 𝓗)

1. **Familiar-task equivalence** at h = H: paired TOST on NLL(E_ID),
   NLL(E_ID^aff) and Acc(E_ID), at the margins ε_m. This is the precondition
   for an afterimage claim.
2. The **primary endpoint** τ(H) on G.
3. The **secondary family** (§8.2).
4. The **persistence curve** τ(h), h ∈ 𝓗, which is descriptive.

Comparisons are made only at **fixed step budgets**. All arms have taken
exactly T0+K+h steps. A "matched familiar-task loss" comparison (each arm at
the first checkpoint where its ID loss crosses a threshold) is reported only
as description. It conditions on a post-treatment outcome, so it carries no
inferential weight.

### 7.4 Separating residue from incomplete correction

| Observation | Reading |
|---|---|
| ID equivalence fails at H | No afterimage claim is possible. The result is "correction incomplete at H" (category D). |
| ID equivalence holds, τ_aff > 0 on G but τ_inv ≈ 0 | The difference is concentrated where the false rule would change the answer. Most consistent with residual rule application on new compositions: a local, rule-specific residue. |
| ID equivalence holds and τ_inv ≠ 0 | The difference appears where applying the false rule end to end cannot explain it: a change in how compositions are computed. This is the strongest evidence for an afterimage that is more than leftover rule use. |
| τ_X ≠ 0, τ_¬X ≈ 0 | The residue is local to compositions involving the target operator. |
| τ_X ≈ τ_¬X ≠ 0 | A global shift in generalization. Check S and N to rule out generic perturbation. |
| τ(h) shrinks steadily toward 0 over 𝓗 | Delayed recovery. Persistence is only claimed for h ≤ H. |

### 7.5 The control arms

* **N, nuisance placebo.** Correct labels, with a different data order inside W.
  E[Y^N − Y^C] = 0 by exchangeability. It is used to:
  1. Validate the pipeline: the primary analysis applied to N−C must not
     reject. Rejection at p < 0.01 marks the run invalid (category D). The
     stricter threshold keeps a 5% chance of a false alarm from voiding a
     sound experiment.
  2. Set the size of generic perturbation, which feeds the dispersion
     estimand and the margins.
* **S, matched incoherent noise.** The same number of wrong labels per batch
  as I, but with no rule behind them. I − S isolates *coherence*: a learnable
  false rule versus label noise the model cannot learn.
* **Not included**, with reasons:
  * A "frozen-for-K-steps" arm: it changes the step count and the LR schedule.
  * A "drop X examples" arm: it changes inputs.
  * A one-bit initialization perturbation: N already covers generic
    perturbation, at a matched duration.

---

## 8. Behavioral evaluation methodology

### 8.1 Primary endpoint

τ(H) is the paired difference in **mean answer NLL on `ood_conf`**, in nats,
I minus C, at a fixed horizon H.

**Why NLL.** It is a proper scoring rule, it is continuous, and it is the
quantity being optimized. It picks up shifts in confidence that leave accuracy
unchanged (§1.5, item 1). It is also on the same scale as the ID equivalence
metric, which makes leakage arguments quantitative.

**Known weakness.** NLL has a heavy tail: a few confident errors dominate it.
The pre-registered robustness check repeats the primary analysis on the Brier
score. If NLL and Brier disagree in sign, that is reported prominently.
Accuracy on `ood_conf` is reported for every comparison but is never the test
statistic.

### 8.2 Secondary family (Holm-corrected together, α = 0.05)

| ID | Estimand | Question |
|---|---|---|
| S1 | τ on G_X | Is the effect in compositions that involve X? |
| S2 | τ on G_¬X | Does it spread to compositions without X? |
| S3 | τ on G_inv | Does it survive where the false rule cannot explain an error? |
| S4 | E[Y^I − Y^S] on G | Is it specific to a coherent rule, beyond matched noise? |
| S5 | κ = E[D_b(I,C) − D_b(N,C)] on G | Does the false rule cause more functional divergence than an innocuous perturbation of the same length, whatever the sign? |
| S6 | ρ, savings (§8.3) | Is there latent retention that behavior does not show? |

### 8.3 Relearning savings

**Why it is justified.** Relearning speed is the most sensitive behavioral
probe of latent retention known: relearning attacks recover "unlearned"
knowledge that behavioral metrics had certified as gone (Hu et al. 2024; Deeb
& Roger 2024; Hinton & Plaut 1987).

**Protocol.** From each arm's state at T0+K+H, fork R = 300 `⟨pilot⟩` steps of
false-rule supervision. The batches are the W batches, replayed with indices
shifted by a fixed offset so they are the same for every arm. The constant
learning rate is η. AUC_a is the area under the false-label NLL curve on
`id_dev^aff`, evaluated every 10 steps. The estimand is ρ = E[AUC_C − AUC_I],
where a positive value means I re-adopts XNOR faster. N gives the reference
spread.

**Caveat.** Savings show that something which *makes re-adoption easier*
remains. They do not show that the false rule is stored in any particular
form (§4.3, point 2).

### 8.4 Descriptive diagnostics, with no inferential claims

* Error rate by depth, broken out for each held-out triple.
* Per-example paired NLL differences: the distribution and how many are
  extreme.
* `depth5` accuracy and NLL.
* The persistence curve τ(h).
* The Brier-score robustness check.
* The matched-ID-loss comparison.

### 8.5 Protecting the confirmatory test set

* The `ood_conf` and `id_conf` seeds are **derived from the git commit hash of
  the frozen pre-registration**:
  `seed = int(sha256(b"ood_conf:" + commit_hash).hexdigest()[:16], 16)`.
  They cannot exist before the analysis plan is frozen, and anyone can check
  the derivation.
* All development and pilot work uses `ood_dev` and `id_dev` only.
* `afterimage analyze --confirmatory` refuses to run if the working tree is
  dirty, if the config hash differs from the one in the pre-registration, or
  if a results file for `ood_conf` already exists. Confirmatory analysis runs
  once.
* Any example or subset found while looking at data, for example "errors
  cluster on `(I,1,X)` at depth 3," is labeled exploratory. It becomes
  confirmatory only if it is pre-registered for a *new* run with a *new*
  sealed test set.

---

## 9. Statistical analysis and acceptance criteria

### 9.1 Pilot (M5)

* 8 blocks, using all arms, `ood_dev` and `id_dev` only, and a separate master
  seed.
* Fixes: T0, K, H, the gate thresholds, ε_m and δ_G (§2.7).
* Estimates: σ_d, the standard deviation of the per-block paired difference d_b
  = Y_b^I − Y_b^C. The *effect* estimates from the pilot are reported as
  exploratory and are never pooled with confirmatory data.

### 9.2 Sample size

The sample size is set so the study can tell detection apart from
equivalence. For a two-sided paired test with power 1−β against an effect of
size δ_G:

  n_det = ⌈((z_{1−α/2} + z_{1−β}) · σ_d / δ_G)²⌉

For TOST power 1−β when the true τ = 0:

  n_eq = ⌈((z_{1−α} + z_{1−β/2}) · σ_d / δ_G)²⌉

Take n = max(n_det, n_eq, 16), computed with
`statsmodels.stats.power.TTestPower().solve_power(...)` and its t-distribution
correction.

**Worked example (illustrative numbers):** with σ_d = 0.02 nats, δ_G = 0.01
nats, α = 0.05 and power 0.8, n_det ≈ 32 and n_eq ≈ 35, so n = 35. If the
pilot implies n > 128, the margins or design are revisited *before*
confirmatory runs. The study is not run underpowered.

### 9.3 Primary analysis

```python
d = Y_I - Y_C                                                       # (n,) one value per block
t_res  = scipy.stats.ttest_1samp(d, 0.0)                            # primary test
ci_t   = t_res.confidence_interval(0.95)                            # primary CI
ci_bca = scipy.stats.bootstrap((d,), np.mean, confidence_level=0.95,
                               n_resamples=100_000, method="BCa",
                               random_state=ANALYSIS_SEED).confidence_interval
w_res  = scipy.stats.wilcoxon(d)                                    # robustness (sign-rank)
tost_p, *_ = statsmodels.stats.weightstats.ttost_paired(Y_I, Y_C, -delta_G, delta_G)
effect = d.mean(); dz = d.mean() / d.std(ddof=1)                    # report both raw and standardized
```

The primary inference is the paired t-test and its CI. With n ≥ 16 and
per-block means over 20k examples, the CLT approximation is reasonable. The
BCa bootstrap and Wilcoxon results are reported next to it. If they disagree
with the t-test about whether zero is excluded, that is reported, and the
claim falls back to the more conservative result.

### 9.4 Secondary analyses and multiplicity

The six p-values from S1–S6 go through
`statsmodels.stats.multitest.multipletests(p, alpha=0.05, method="holm")`.
Every estimate is reported with its unadjusted 95% CI and its Holm-adjusted p.
The primary endpoint has α = 0.05 to itself and is not part of the family.
Exploratory analyses get estimates and intervals, but no claim of
significance.

### 9.5 Sensitivity analyses (pre-registered and reported, not confirmatory)

* A crossed bootstrap over blocks × examples, which brings test-set sampling
  into the uncertainty.
* Brier score in place of NLL.
* Excluding blocks that failed Stage B (per-protocol), shown next to the
  intention-to-treat result.
* Worst-case imputation for blocks that failed after the fork.

### 9.6 Outcome categories (pre-registered decision rules)

| Category | Rule |
|---|---|
| **A. Persistent difference detected** | Validity holds, ID-equivalence TOST passes for every ID metric, and the 95% CI for τ(H) excludes 0. A1 if |τ̂| ≥ δ_G ("practically meaningful"). A2 if the 90% CI also lies inside ±δ_G ("detected, but too small to matter"). |
| **B. Practical equivalence** | Validity holds, ID equivalence passes, the TOST at δ_G rejects, and the 95% CI includes 0. |
| **C. Inconclusive** | Validity holds and ID equivalence passes, but neither A nor B applies. Reported with the achieved power and with the n that would be needed. |
| **D. Invalid** | Any of: replay or resume determinism fails on the confirmatory hardware; the Stage A hash check fails; the N−C placebo rejects at p < 0.01; ID equivalence fails at H (reported as "correction incomplete at H"); more than 10% of blocks fail after the fork; leakage is detected. |

The results are reported whatever the category. A D outcome is still written
up, with its diagnosis.

### 9.7 Failed and unstable runs

* **Before treatment** (gate failure at T0): exclude and replace, and report
  the count. This is valid because no arm has been treated yet.
* **After the fork** (NaN, divergence, crash that cannot be resumed):
  * Never silently dropped.
  * Crashes are resumed from the last checkpoint that verifies. Determinism
    makes the resumed run identical.
  * A real numerical divergence is a *result* of that arm: it enters the
    sensitivity analysis with worst-case imputation and counts toward the 10%
    limit.
  * Blocks are never excluded because they failed to recover (§0.1).

---

## 10. Training-dynamics and representation diagnostics

Everything in this section is **secondary and descriptive** unless a
statistic is listed in §8.2. Behavioral results come first, and nothing here
can promote or demote the outcome category.

### 10.1 What is recorded, and how often

| Quantity | Cadence | Storage |
|---|---|---|
| Training loss, per-batch affected rate, LR | every step | `steps.parquet` |
| Per-tensor gradient norm, update norm ‖Δθ‖/‖θ‖ | every 10 steps | `dyn.parquet` |
| Adam ν statistics per tensor (mean, p50, p99, max) | every 100 steps | `dyn.parquet` |
| Gradient cosine between arms on the *same* correction batch | every 100 steps (recompute ∇ for arm C at the stored θ_I and θ_C) | `xarm.parquet` |
| Full answer distributions on `probe` (4,096 × 2 log-probs) | at every checkpoint | `probe_logits/<step>.parquet` |
| Residual-stream activations at selected positions (the X token, EQ) per layer on 512 probes | at every checkpoint | `acts/<step>.safetensors` |
| ID/OOD metrics, with all subsets | at every checkpoint | `eval.parquet` |
| Wall time, GPU utilization, memory | every 10 s | `gpu.parquet` (NVML) |

Checkpoints: 20 log-spaced points over [0, T0], then T0, T0+K/4, T0+K/2, T0+K,
then T0+K+h for h ∈ 𝓗. Full training state is kept at T0, T0+K and T0+K+H.
Weights only are kept elsewhere.

### 10.2 Comparing trajectories without overreading parameters

* **Function first.** D(I,C) and D(N,C) are compared on the same probes over
  time. The question is whether the false rule causes more functional
  divergence, and divergence that lasts longer, than an innocuous perturbation
  of the same length. Statistic S5 answers this at H.
* **Parameter distances only next to the N reference.** ‖θ_I − θ_C‖ means
  nothing on its own: chaotic divergence makes it grow even for N. Only the
  ratio to ‖θ_N − θ_C‖ is interpreted.
* **Linear-interpolation barrier** between I and C, and between N and C, at H.
  A near-zero barrier means the arms share a basin, which also makes
  cross-model activation patching (§10.4) meaningful. A barrier that appears
  only for I suggests the false rule moved the model into a different basin,
  which is the Juneja et al. (2023) style of explanation.
* **Changed later learning dynamics.** Do correction-phase gradients in I
  point the same way as C's on identical batches? Does the gradient cosine
  approach that of N, and how fast? Does ν in I stay inflated, which would
  mean smaller effective steps for a long time?

### 10.3 CKA, and what it cannot show

* Linear CKA (debiased HSIC) per layer, at the X-token and EQ positions, on
  fixed probes, for (I, C) and (N, C), at T0+K and at H.
* **Assumptions:** CKA is invariant to orthogonal transformations and
  isotropic scaling. It compares the geometry of a probe sample, so it depends
  on the probe distribution.
* **Limitations:** it is sensitive to outliers and can be moved without any
  functional change (Davari et al. 2023). It can miss changes that matter
  functionally (Ding et al. 2021).
* **Interpretation rule:** a CKA difference between I and C counts as
  "aligned with behavior" only if it is outside the N–C envelope *and* its
  layer profile agrees with the patching result in §10.4. It never counts as
  evidence of retention on its own.

### 10.4 Activation patching across forks (the strongest causal diagnostic)

Because I and C share a fork and, if the barrier is near zero, a basin, a
residual-stream activation can be moved from C into I at layer ℓ and position
p. We then measure the fraction of the I–C held-out NLL gap that closes:
interchange interventions (Geiger et al. 2021). This is a causal claim, but it
is about *activations*, given the inputs. It localizes where the difference is
mediated. It does not establish a full mechanism.

**Assumptions:** the two models' bases are comparable, which is checked by
the barrier and by patching N into C as a control. The effects of
patching different sites may not add up.

### 10.5 Decomposing the mechanism: optimizer state versus weights

Run two hybrid forks at T0+K:

* (θ_I, μ_C, ν_C): I's weights with C's optimizer moments.
* (θ_C, μ_I, ν_I): C's weights with I's optimizer moments.

Both then run the standard correction. If (θ_I, opt_C) reproduces τ(H) and
(θ_C, opt_I) does not, the residue is carried by the weights. The reverse
would mean it is carried by Adam's statistics, which would be a mundane but
important mechanism. These runs are cheap (two extra arms) and exploratory.

### 10.6 TracIn-style attribution

The TracIn-CP estimate is Σ_{c} η_c ⟨P_c ∇ℓ(θ_c; z), ∇ℓ(θ_c; z′)⟩ over the
stored checkpoints c, where P_c is Adam's diagonal preconditioner
(1/(√ν̂_c + ε)), because plain TracIn assumes SGD. It is used for two things:

1. Attributing the closing of the I–C gap to individual correction examples.
2. As a check on attribution itself: the fork design gives the *exact* effect
   of the whole window W. Comparing TracIn's summed influence of W with the
   measured τ_b shows how far a first-order method can be trusted here.

**Limitations:** first-order only, sparse checkpoints, and Adam's
nonlinearity (Pruthi et al. 2020; Basu et al. 2021; Guu et al. 2023).

### 10.7 Mechanistic hypotheses to weigh against each other

* **H-local:** a residual X-specific computation, predicting τ_aff ≫ τ_inv ≈ 0
  and patching concentrated at the X-token positions.
* **H-route:** the false rule rewired how operands are routed. Predicts τ_inv
  ≠ 0 and effects at the attention layers.
* **H-optimizer:** the residue comes from inflated ν (§10.5).
* **H-basin:** the arms end up in different basins (§10.2).
* **H-null-generic:** I behaves like N. Predicts S5 ≈ 0 and S4 ≈ 0.

If a behavioral effect is found but none of these hypotheses is supported,
the report says exactly that: *effect established, mechanism unknown*.

---

## 11. GPU infrastructure and cost

### 11.1 Throughput estimates, to be replaced at M2

These are order-of-magnitude estimates only. The steps-per-second figure must
be measured in M2 and recorded in `docs/measured_throughput.md` before the
scaling plan below is used.

| Item | Estimate |
|---|---|
| Step time (0.8M parameters, B = 256, FP32, eager, deterministic) | ~3–6 ms on a current datacenter GPU; ~30–80 ms on CPU |
| Base phase T0 = 30k steps `⟨pilot⟩` | ~2–3 min GPU |
| One arm, K + H = 22k steps, plus evaluation | ~1.5–2.5 min |
| One block (base + 4 arms + savings probes) | ~10 min |
| Confirmatory run, 35 blocks | **~6 GPU-hours** |
| Pilot, sensitivity analyses and hybrid forks | ~10 GPU-hours |
| Storage | ~10 MB per full state; ~150 MB per block with logs; ~6 GB in total |

### 11.2 What is and is not needed

The model is launch-latency bound and uses a small fraction of a GPU. The
efficient layout is several block processes on one GPU. Four to eight
processes per device is typical, using MPS if it is available. Determinism is
per process, so it is unaffected.

* **Needed:** 1 GPU, and up to 8 on one node to shorten wall-clock time.
* **Not needed:** 256 GPUs, multiple nodes, DDP or FSDP. DDP would also add
  nondeterministic all-reduce ordering and change what a batch means.
* `torch.func.vmap` ensembling of arms was considered and rejected: it saves
  time but complicates the determinism and checkpoint contracts.
* Scale-out is justified only by measured throughput. GPUs needed =
  ⌈total GPU-hours / (target wall-clock hours × processes per GPU ×
  per-process efficiency)⌉.

### 11.3 Ray

Ray is not justified. The jobs are independent, coarse (one block per job),
idempotent and resumable from files. A ~100-line launcher handles them:

1. A job list (block IDs).
2. One `fcntl` lock file per job on shared storage.
3. N worker processes per GPU pinned with `CUDA_VISIBLE_DEVICES`.
4. Each worker claims a free lock, runs `afterimage run-block`, and releases
   the lock.

Ray adds a runtime, a scheduler and failure modes that do nothing for this
workload. Reconsider it only if the job count exceeds ~10⁴ or work must
span several nodes, and neither is expected.

### 11.4 Software and container

* `pyproject.toml` with `uv.lock`, and `uv sync --frozen` in CI and in the
  container.
* The container is built from an official PyTorch CUDA runtime image, **pinned
  by digest** (`image@sha256:…`). Tags are mutable, so tags alone are not
  enough. The exact digest, PyTorch version and CUDA version are fixed at M2
  and recorded in `docs/environment.md`.
* The same image runs CPU tests in CI.

### 11.5 Experiment metadata (`RunMeta`, written with every checkpoint)

* The git commit, and a dirty flag. Confirmatory runs refuse to start on a
  dirty tree.
* The config file's SHA-256, the dataset manifest's SHA-256, and the image
  digest.
* `torch.__version__`, `torch.version.cuda`, `torch.backends.cudnn.version()`,
  the GPU name and driver version (NVML), the CPU model, and the values of
  the determinism flags.
* Host identity as a salted hash, so audits can group runs by host without
  recording host names.

### 11.6 Recovery

Every stage writes a completion marker only after its checkpoint hashes
verify. `run-block` is idempotent: it finds the latest verified checkpoint
for each arm and resumes from there. The stateless data schedule (§6.3) and
the saved optimizer and scheduler state make resumption exact, which the
resume test in §6.5 checks.

### 11.7 Monitoring and accounting

* An NVML sampler thread records GPU utilization, memory and power every 10
  seconds to `gpu.parquet`.
* Each job appends one row to `ledger.parquet`: job, block, arm, GPU type,
  start time, end time, GPU-seconds, exit status.
* `afterimage ledger` totals GPU-hours by phase.

### 11.8 Shutting down and controlling cost on an on-demand provider

* The provisioning wrapper provisions **one node by default**. `--max-gpus` is
  required to go above 1.
* The launcher computes the projected cost as ledger total + remaining jobs ×
  measured cost per job. It refuses to start if the projection exceeds
  `--budget-gpu-hours`. It also stops claiming jobs once the budget is spent.
* When the job queue is empty, the node calls the provider's terminate API.
  The call is provider-specific and lives in one script. As a backstop,
  `shutdown -h +15` is scheduled at boot and renewed only while jobs are
  running.
* Checkpoints live on shared storage, so terminating a node loses no work.

### 11.9 Three kinds of reproducibility

| Kind | What it means here | Expected? |
|---|---|---|
| **Deterministic execution** | The same process, inputs and hardware give the same bits on every run | Yes. Required and tested (§6.5). |
| **Bitwise reproducibility across machines** | The same GPU model, driver, image and flags give the same bits | Expected but not guaranteed. Checked in replication tier R1 (§14). |
| **Across GPU types** | A different architecture or kernel selection | **Not expected** to be bitwise. Floating-point reductions differ. |
| **Statistical reproducibility** | Fresh blocks, possibly on other hardware, give the same outcome category and overlapping CIs | Required. This is what the scientific claim rests on. |

All arms of a block run on one device in one process, so the paired
difference never includes hardware noise. Blocks may run on different
devices, and device type is recorded as a covariate.

---

## 12. Repository architecture and key interfaces

```
afterimage/
├── pyproject.toml / uv.lock
├── configs/
│   ├── smoke.toml            # CPU, 2 blocks, tiny T0/K/H; runs in CI
│   ├── pilot.toml
│   └── confirmatory.toml     # frozen with the preregistration
├── prereg/preregistration.md # analysis plan; its commit hash seeds the sealed sets
├── src/afterimage/
│   ├── lang/                 # grammar.py, interp.py, interp_stack.py, generate.py,
│   │                         # intervention.py, signature.py, enumerate.py
│   ├── data/                 # build.py (splits → Parquet + manifest), schedule.py (§6.3)
│   ├── model/                # transformer.py
│   ├── train/                # state.py (TrainState I/O, fork), loop.py, determinism.py, gate.py
│   ├── experiment/           # design.py (arms, windows, horizons), run_block.py, launcher.py
│   ├── eval/                 # metrics.py, probes.py, stages.py (A/B/C checks)
│   ├── analysis/             # endpoints.py, stats.py, power.py, report.py
│   ├── diagnostics/          # cka.py, interpolation.py, patching.py, tracin.py, hybrids.py
│   ├── artifacts/            # hashing.py, manifest.py, env.py, ledger.py
│   └── cli.py                # `afterimage …` entry point (argparse)
├── scripts/reproduce.sh
└── tests/                    # mirrors src/; `slow` and `gpu` markers
```

### 12.1 Dependencies and why each is there

| Dependency | Why | Alternative rejected |
|---|---|---|
| Python ≥ 3.11 | `tomllib` is in the standard library | — |
| PyTorch | Model, autograd, AdamW, the determinism switches | JAX: excellent, but a mixed ecosystem gains nothing here |
| NumPy | Generator, keyed RNG, enumeration | — |
| safetensors | Checkpoint loading without pickle; easy to audit | `torch.save`: pickle is unsafe for auditors and less portable |
| pyarrow (Parquet) | Typed, columnar, compressed tables for results and the pool; readable from any language | CSV loses types and is huge; a database is unnecessary |
| SciPy | `ttest_1samp`, `wilcoxon`, `bootstrap` (BCa) | — |
| statsmodels | `ttost_paired`, `multipletests`, `TTestPower` | Hand-rolled versions invite bugs |
| pytest + Hypothesis | Exhaustive and property-based oracle tests | — |
| pynvml | GPU sampling for accounting | Parsing `nvidia-smi` output is brittle |
| *Not used* | Ray, Hydra, W&B, MLflow, any database, web dashboards | They add complexity without answering the question. Logs are files, and plots are scripts. |

### 12.2 Core interfaces

```python
# lang/grammar.py
class Op(IntEnum): N = 5; A = 6; O = 7; I = 8; X = 9         # == token ids
@dataclass(frozen=True, slots=True)
class Tree:
    op: Op | None             # None for constants
    value: int | None         # 0/1 for constants
    kids: tuple["Tree", ...] = ()
def to_postfix(t: Tree) -> list[int]: ...
def from_postfix(toks: Sequence[int]) -> Tree: ...

# lang/interp.py
class Semantics(Enum): TRUE = 0; FALSE_XNOR = 1
def evaluate(t: Tree, sem: Semantics = Semantics.TRUE) -> int: ...
def affected(t: Tree) -> bool: return evaluate(t, TRUE) != evaluate(t, FALSE_XNOR)

# lang/signature.py
Triple = tuple[Op, int, Op]
def signature(t: Tree) -> frozenset[Triple]: ...

# data/build.py
@dataclass(frozen=True)
class SplitSpec: name: str; depths: tuple[int, ...]; heldout: Literal["exclude", "require"]; n: int; seed_key: str
def build_splits(cfg: DataConfig, out: Path) -> Manifest: ...   # writes <split>.parquet + manifest.json

# Parquet columns (every split)
#   tokens: fixed_size_list<uint8>[40] (PAD-filled); eq_pos: uint8; depth: uint8
#   y_true: uint8; y_false: uint8; has_x: bool; sig_mask: uint64 (bit per triple)
#   heldout_x: bool; heldout_nonx: bool; canon_sha1: fixed_size_binary[20]

# experiment/design.py
@dataclass(frozen=True)
class ArmSpec:
    name: Literal["C", "I", "N", "S", "HYB_wI_oC", "HYB_wC_oI"]
    kind: Literal["C", "I", "N", "S"]
    window: range                      # [T0, T0+K)
@dataclass(frozen=True)
class Design: T0: int; K: int; H: int; horizons: tuple[int, ...]; R: int; arms: tuple[ArmSpec, ...]

# analysis/endpoints.py
@dataclass(frozen=True)
class Endpoint: name: str; split: str; subset: str | None; metric: Literal["nll", "brier", "acc"]
def block_values(results: pa.Table, ep: Endpoint, arm: str, h: int) -> np.ndarray: ...   # one per block

# analysis/stats.py
@dataclass
class PairedResult: estimate: float; ci95: tuple[float, float]; ci_bca: tuple[float, float]
                    p: float; p_wilcoxon: float; p_tost: float | None; dz: float; n: int
def paired(y_a: np.ndarray, y_b: np.ndarray, margin: float | None, seed: int) -> PairedResult: ...
def classify(primary: PairedResult, id_equiv: list[PairedResult], validity: Validity) -> Category: ...
```

### 12.3 Configuration example (`configs/confirmatory.toml`)

```toml
[meta]
prereg_commit = "<filled by `afterimage prereg freeze`>"
master_seed   = 20261118

[data]
pool_size = 4_000_000
train_depths = [1, 2, 3, 4]
heldout_x    = [["I",1,"X"], ["A",2,"X"], ["X",1,"O"]]
heldout_nonx = [["I",2,"A"], ["O",1,"I"], ["A",1,"N"]]

[model]
layers = 4; d_model = 128; heads = 4; d_ff = 512; ctx = 40; tie_embeddings = true

[optim]
lr = 5e-4; warmup = 1000; betas = [0.9, 0.98]; eps = 1e-8; weight_decay = 0.01
batch = 256; final_decay_frac = 0.2; final_lr_frac = 0.1; grad_clip = "none"

[design]
n_blocks = 35          # from power analysis
T0 = 30_000            # ⟨pilot⟩
K  = 2_000             # ⟨pilot⟩
H  = 20_000            # ⟨pilot⟩
horizons = [0, 1250, 2500, 5000, 10000, 20000]
R = 300
arms = ["C", "I", "N", "S"]

[analysis]
alpha = 0.05
epsilon = { id_nll = "<pilot>", id_aff_nll = "<pilot>", id_acc = 0.002 }
delta_G = "<pilot>"
placebo_invalid_p = 0.01
```

---

## 13. Development roadmap

Each milestone below lists its objective, what gets built, inputs and outputs
(I/O), tests, compute, acceptance criteria, failure conditions, what a result
means, and what must not be built yet. Milestones are ordered by scientific
dependency.

### M1: Validate the grammar and the oracle

* **Objective:** a ground truth that can be trusted.
* **Build:** `lang/*`, `data/build.py`, `data/schedule.py`.
* **I/O:** a config goes in. Parquet splits and `manifest.json` (hashes,
  counts, affected rates, label marginals under both semantics) come out.
* **Tests:** everything in §4.6, plus a test that rebuilding gives identical
  hashes.
* **Compute:** CPU only, minutes. The exhaustive depth-3 test takes about 1–3
  minutes vectorized.
* **Accept when:** all tests pass. The manifest reports the affected rate
  (expected to be a sizeable minority of depth-≥2 examples) and the size of
  the label-marginal shift.
* **Fail if:** the two oracles disagree, or leakage is found.
* **Means:** the data-generating process is correct. This says nothing yet
  about learnability.
* **Not yet:** any model code.

### M2: A Transformer baseline that learns the task

* **Objective:** clear the learnability gate with the smallest model.
* **Build:** `model/`, `train/loop.py`, `train/gate.py`, the determinism
  setup, metrics on `id_dev` and `ood_dev`, a throughput benchmark.
* **I/O:** base checkpoints and gate reports for 3 seeds.
* **Tests:**
  * Shape and causality: changing future tokens never changes earlier logits.
  * Overfitting one batch reaches 100% accuracy.
  * The loss mask covers only the ⟨EQ⟩ position.
  * The parameter count matches the specification.
  * A one-step CPU run is deterministic.
* **Compute:** under 1 GPU-hour.
* **Accept when:** at least 2 of 3 seeds pass the gate, possibly after the
  escalation ladder (§5.3). Throughput is measured and written down.
* **Fail if:** the ladder is exhausted. Then the next step is to report
  non-learnability and stop.
* **Means:** the model composes well enough to have a held-out generalization
  that can be perturbed.
* **Not yet:** forks or interventions.

### M3: Exact checkpoint forks

* **Objective:** prove that counterfactual histories are exact.
* **Build:** `train/state.py` (save, load, fork), `artifacts/*`,
  `run_block.py` covering only the C arm.
* **Tests:**
  * Save/load round trip is bit-identical.
  * Replay test (§6.5.1).
  * Kill-and-resume test (§6.5.2), killing at a random step.
  * Fork-hash test.
  * The stateless schedule gives the same batches before and after a resume.
* **Compute:** under 1 GPU-hour.
* **Accept when:** replay and resume are bit-identical on the target GPU type.
* **Fail if:** they are not. Then find the nondeterministic operation, or
  fall back to the measured C′ nuisance arm and document why.
* **Means:** τ_b is an exact unit-level effect.
* **Not yet:** analysis code, CKA.

### M4: Confirm the intended effect of false supervision

* **Objective:** Stage B works. The intervention changes exactly what it is
  meant to change.
* **Build:** arms I, N and S; Stage A and Stage B checks; batch-level audit
  logging.
* **Tests:**
  * In I, the inputs of every W batch are byte-identical to C's.
  * Labels differ exactly on A_t.
  * In S, |R_t| = |A_t|.
  * In N, inputs differ only inside W.
  * All arms are identical outside W in both inputs and labels.
  * Running with the I arm's window set to empty reproduces C bit for bit.
* **Compute:** about 1 GPU-hour.
* **Accept when:** the manipulation check passes at the pilot K in at least
  95% of blocks.
* **Fail if:** XNOR is never adopted at reasonable K. That is informative:
  report it, then try a larger K or LR in the window.
* **Means:** the treatment was delivered.
* **Not yet:** looking at held-out outcomes of the I arm beyond τ(0).

### M5: Pilot and pre-registration

* **Objective:** fix every free parameter without touching confirmatory data.
* **Build:** `analysis/power.py`, a margin-calibration script, `afterimage
  prereg freeze` (writes the pre-registration, commits it, derives the sealed
  seeds), and the persistence-curve report.
* **I/O:** 8 pilot blocks on dev splits go in. The pre-registration and
  `confirmatory.toml` come out.
* **Tests:**
  * The analysis pipeline on synthetic data with a *planted* effect: the CI
    covers the truth at the nominal rate over 1,000 simulations, and power
    matches the formula within ±5 percentage points.
  * The placebo pipeline rejects at α under the null.
* **Compute:** about 2 GPU-hours.
* **Accept when:** H is chosen so that ID equivalence (on the dev splits)
  holds in at least 95% of pilot blocks; n ≤ 128; the pre-registration is
  committed.
* **Fail if:** ID equivalence cannot be reached within a feasible H. Report
  "correction incomplete," and choose H larger only if that is affordable.
* **Means:** the design is now frozen.
* **Not yet:** generating `ood_conf`.

### M6: Confirmatory corrective-training runs

* **Objective:** collect the confirmatory data.
* **Build:** `launcher.py`, the ledger, the budget guard.
* **I/O:** `confirmatory.toml` goes in. Blocks × arms × checkpoints and
  `eval.parquet` come out.
* **Tests:** a smoke config runs end to end on CPU in CI; the launcher's
  lock and claim logic is tested with simulated crashes.
* **Compute:** about 6 GPU-hours, ≤ 1 day on 1 GPU.
* **Accept when:** all blocks reach their final verified checkpoint and the
  ledger reconciles.
* **Fail if:** more than 10% of blocks fail after the fork, which is category
  D.
* **Means:** data now exist. No analysis has been run.
* **Not yet:** any confirmatory analysis.

### M7: Familiar-task equivalence and the held-out evaluation

* **Objective:** Stage C on the sealed sets.
* **Build:** generation of `ood_conf` and `id_conf` from the frozen seed;
  `eval/stages.py`.
* **Tests:** the sealed seed derivation matches the pre-registration commit;
  the refusal guards in §8.5 work.
* **Compute:** minutes.
* **Accept when:** evaluation is complete for every block, arm and horizon.
* **Fail if:** leakage is found in the sealed set (category D).
* **Means:** the raw results exist.
* **Not yet:** looking at the diagnostics.

### M8: Uncertainty across seed blocks

* **Objective:** decide the outcome category.
* **Build:** `analyze --confirmatory`, which runs once and produces the
  report.
* **Tests:** golden-file tests for the report on synthetic data; `classify()`
  tested on every category boundary.
* **Compute:** minutes.
* **Accept when:** the report states the category (A1, A2, B, C or D) with
  every estimate and CI.
* **Fail if:** never. Every category is reported.
* **Means:** this is the scientific result.
* **Not yet:** mechanistic claims.

### M9: Representations as secondary diagnostics

* **Objective:** look for explanations, without overclaiming.
* **Build:** `diagnostics/*` (§10).
* **Tests:**
  * CKA of X with itself is 1. CKA is invariant to orthogonal transforms and
    behaves on a reference example.
  * Patching C into C changes nothing.
  * TracIn on a 2-parameter toy matches the analytic result.
  * Interpolation at α = 0 and α = 1 gives the endpoint losses.
* **Compute:** about 2 GPU-hours, mostly the hybrid forks and patching.
* **Accept when:** every diagnostic is reported against the N reference, and
  each mechanistic hypothesis in §10.7 is marked supported, unsupported or
  untested.
* **Fail if:** the diagnostics disagree with each other. Report that, and
  claim nothing.
* **Means:** possible mechanisms, labeled exploratory.
* **Not yet:** new experiments chosen on the basis of these results inside
  this study.

### M10: Package for independent replication

* **Objective:** an outsider can reproduce the work and challenge it.
* **Build:** `scripts/reproduce.sh`, `afterimage audit`, artifact upload,
  `docs/replication.md`, and a dedicated related-work search with an
  honest novelty statement.
* **Tests:** a fresh-machine dry run of `reproduce.sh --tier R1` against the
  released checkpoints.
* **Compute:** about 1 GPU-hour.
* **Accept when:** an R1 replication by someone else, or on a separate
  machine, matches bit for bit.
* **Fail if:** it does not. Fix the packaging before release.
* **Means:** the result can now be checked by others.
* **Not yet, and not in this project:** extensions to other languages,
  operators or scales. Those are follow-up studies with their own
  pre-registrations.

---

## 14. Independent reproduction and audit procedure

### 14.1 Released artifacts

* The source code at a tagged commit, plus `uv.lock` and the container image
  digest.
* `configs/*.toml` and `prereg/preregistration.md`, whose commit hash seeds
  the sealed sets.
* The data generator and oracle, and `manifest.json` with the SHA-256 of every
  split. The splits themselves are released too, though they can be
  regenerated.
* For every block: the fork state at T0, the states at T0+K and T0+K+H for
  every arm, weights at every checkpoint, and the optimizer and RNG state
  inside the full states.
* `steps/dyn/xarm/eval/gpu.parquet`, the probe logits, `ledger.parquet` and
  `RunMeta` for every run.
* The raw per-block, per-example evaluation results.
* The analysis scripts and the frozen analysis output.
* `docs/environment.md` (hardware and software) and `docs/limitations.md`.

### 14.2 Executable procedure (planned CLI)

```bash
git clone <repo> && cd afterimage && git checkout v1.0
uv sync --frozen                         # or: docker run <image>@sha256:<digest>
afterimage data build  --config configs/confirmatory.toml --verify   # hashes must match manifest
afterimage test        --all             # oracle, intervention, determinism suites
# Tier R1 — bitwise, from released fork checkpoints (same GPU type)
afterimage replay      --artifacts <path> --blocks 0-34 --check-bitwise
# Tier R2/R3 — full re-run with fresh hardware (optionally fresh master seed)
afterimage run         --config configs/confirmatory.toml --blocks all
afterimage analyze     --config configs/confirmatory.toml --confirmatory --out results/
afterimage audit       --results results/  # placebo, leakage, hashes, determinism, ledger
```

### 14.3 What counts as replication

| Tier | Procedure | Success | Partial | Failure |
|---|---|---|---|---|
| R1 bitwise | Replay from released fork states on the same GPU model and image | Every final-state hash matches | Hashes differ, but every per-block τ_b is within 10⁻⁶ | Per-block results differ materially |
| R2 same-design rerun | Full rerun with the released seeds on equivalent hardware | Same category, CI overlap ≥ 50% | Same sign, different category because of power | Different sign, or category A ↔ B |
| R3 independent | Fresh master seed, other hardware, possibly an independent reimplementation from this blueprint | Same category, and the original τ̂ is inside the replication's 95% CI | Same sign, and the CIs overlap | The replication's 95% CI excludes the original τ̂ *and* the categories differ |

### 14.4 Tests that prove the intervention changes only its target

These are in §4.6 and M4: byte equality of inputs, label difference exactly on
A_t, the empty-window run equal to C, and arms identical outside W. They run
in CI on every commit.

### 14.5 How an auditor could challenge the causal reading, and the checks
each challenge maps to

1. **"The held-out set isn't compositional."** Check pool signatures against
   ℋ with `afterimage audit leakage`, and inspect the triple definitions.
2. **"The false rule never took hold."** Inspect Stage B per block.
3. **"Correction was incomplete."** Inspect the TOST on NLL(E_ID^aff), the
   G_inv results, and the persistence curve.
4. **"It's generic perturbation."** Compare against N and S, and look at S5.
5. **"The test set was tuned on."** Verify that the sealed seed comes from the
   pre-registration commit, check git timestamps, and confirm the
   confirmatory analysis ran once (its log is in the artifacts).
6. **"The margins were chosen after the fact."** The margins are in the
   pre-registration commit, which predates `ood_conf`.
7. **"Numerical noise."** Rerun R1, and inspect the replay logs.
8. **"Cherry-picked horizon."** H is pre-registered, and the full τ(h) curve
   is released.
9. **"Swap the arm labels."** The analysis is symmetric. Recomputing with C
   and I swapped must flip the sign exactly, and `audit` checks this.
10. **"An alternative explanation: optimizer state, a different basin."**
    These are the §10.5 hybrids and §10.2 barriers, released as exploratory
    results.

---

## 15. Possible findings, negative results, confounds and limits

### 15.1 Possible outcomes and what each would mean

* **A1, with τ_inv ≠ 0, S4 > 0 and S5 > 0.** A coherent false rule leaves a
  rule-specific residue in compositional generalization that goes beyond leftover
  rule use and beyond generic perturbation, within horizon H. This is the
  strongest positive case, and it contrasts with the label-permutation
  recovery reported by Achille et al. (2019).
* **A1 with τ_inv ≈ 0.** The residue is confined to compositions where the
  false rule changes the answer: local, rule-level persistence. It is less
  surprising, and still a measured boundary.
* **B together with S6 > 0.** Behavior has recovered on new compositions, yet
  re-adoption is faster: "behaviorally silent latent residue." This agrees
  with the relearning results in the unlearning literature, and adds a
  controlled, oracle-backed, from-scratch instance with a placebo.
* **B together with S6 ≈ 0 (equivalence).** Within the tested conditions, a
  temporary coherent false rule leaves no detectable behavioral or relearning
  trace once familiar-task behavior has recovered. This is a well-bounded
  negative result, and as valuable as a positive one.
* **C.** Report the achieved power and the n that would be needed.
* **D, from correction being incomplete at H.** This is itself a finding: in
  this setting, correcting a coherent false rule takes longer than H at the
  given learning rate.

### 15.2 Confounds that remain after the design

* **Effective exposure.** I has fewer correct X-signals. This is inseparable
  from the treatment, and S only matches the *count* of wrong labels.
* **Complementarity of XNOR.** The false rule is a negation of the true one,
  so the model could adapt by learning a global flip. That could make the
  intervention unusually easy to undo, or unusually easy to re-adopt, which
  limits generalizing to other false rules (for example XOR → OR, a 1-row
  change, or one with a different marginal). That is a follow-up, not part of
  this study.
* **Margins are conventions.** ε and δ_G come from pilot variability. Someone
  who uses a different SESOI can reread the CIs.
* **Conditioning on the pool and test set.** Primary inference is conditional
  on one training pool and one sealed test set. The crossed bootstrap
  addresses the test set only partly, and the pool not at all.

### 15.3 Limits on scope

* One language, one target operator, one false rule, one optimizer, one model
  size, one schedule, and a finite H. Every claim is "under design 𝒟."
* A model trained from scratch on a synthetic task, at 0.8M parameters.
  Nothing here transfers automatically to pretrained or large models, in
  either direction.
* Systematicity only. Productivity (`depth5`) is exploratory and probably at
  the floor.
* Behavioral recovery, functional equivalence, representational similarity
  and erasure of information are four different things:
  * Behavioral equivalence on the tested distributions is the most this study
    can establish.
  * Savings can show that information is *not* erased.
  * Neither this study nor any weight-based measurement can certify erasure
    (Thudi et al. 2022).
* Representational diagnostics describe. Only behavioral endpoints and
  activation patching support causal claims, and patching only at the level
  of activations.

---

## References

Entries marked † were checked against the publisher or arXiv page when this
blueprint was written. The others are cited from the author's working
knowledge and must be verified in M10 before any manuscript cites them.

**Training dynamics, stability, path dependence**
- Achille, Rovere, Soatto. *Critical Learning Periods in Deep Networks.* ICLR 2019. arXiv:1711.08856 †
- Frankle, Dziugaite, Roy, Carbin. *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* ICML 2020. arXiv:1912.05671
- Fort, Dziugaite, Paul, Kharaghani, Roy, Ganguli. *Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel.* NeurIPS 2020. arXiv:2010.15110
- Summers, Dinneen. *Nondeterminism and Instability in Neural Network Optimization.* ICML 2021. arXiv:2103.04514 †
- Ash, Adams. *On Warm-Starting Neural Network Training.* NeurIPS 2020. arXiv:1910.08475
- Nikishin, Schwarzer, D'Oro, Bacon, Courville. *The Primacy Bias in Deep Reinforcement Learning.* ICML 2022. arXiv:2205.07802
- Lyle, Zheng, Nikishin, Pires, Pascanu, Dabney. *Understanding Plasticity in Neural Networks.* ICML 2023. arXiv:2303.01486
- Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton. *Loss of plasticity in deep continual learning.* Nature 632, 2024.
- Wu, Dyer, Neyshabur. *When Do Curricula Work?* ICLR 2021. arXiv:2012.03107
- Dodge et al. *Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping.* 2020. arXiv:2002.06305

**Underspecification and out-of-distribution variance**
- McCoy, Min, Linzen. *BERTs of a Feather Do Not Generalize Together.* BlackboxNLP 2020. arXiv:1911.02969
- D'Amour et al. *Underspecification Presents Challenges for Credibility in Modern Machine Learning.* JMLR 2022. arXiv:2011.03395
- Juneja, Bansal, Cho, Sedoc, Saphra. *Linear Connectivity Reveals Generalization Strategies.* ICLR 2023. arXiv:2205.12411 †

**Forgetting, unlearning, relearning, persistence**
- McCloskey, Cohen. *Catastrophic Interference in Connectionist Networks.* Psychology of Learning and Motivation 24, 1989.
- Hinton, Plaut. *Using Fast Weights to Deblur Old Memories.* Proc. Cognitive Science Society, 1987.
- Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks.* PNAS 2017. arXiv:1612.00796
- Toneva et al. *An Empirical Study of Example Forgetting during Deep Neural Network Learning.* ICLR 2019. arXiv:1812.05159
- Jain, Kirk, Lubana, Dick, Tanaka, Grefenstette, Rocktäschel, Krueger. *Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks.* ICLR 2024. arXiv:2311.12786 †
- Kotha, Springer, Raghunathan. *Understanding Catastrophic Forgetting in Language Models via Implicit Inference.* ICLR 2024. arXiv:2309.10105
- Hu, Fu, Wu, Smith. *Jogging the Memory of Unlearned LLMs Through Targeted Relearning Attacks.* NeurIPS 2024 (earlier: ICML 2024 workshop). arXiv:2406.13356 †
- Lynch, Guo, Ewart, Casper, Hadfield-Menell. *Eight Methods to Evaluate Robust Unlearning in LLMs.* 2024. arXiv:2402.16835
- Deeb, Roger. *Do Unlearning Methods Remove Information from Language Model Weights?* 2024. arXiv:2410.08827 †
- Thudi, Jia, Shumailov, Papernot. *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning.* USENIX Security 2022. arXiv:2110.11891 †
- Hubinger et al. *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* 2024. arXiv:2401.05566
- Zhang, Bengio, Hardt, Recht, Vinyals. *Understanding deep learning requires rethinking generalization.* ICLR 2017. arXiv:1611.03530

**Training-data attribution**
- Koh, Liang. *Understanding Black-box Predictions via Influence Functions.* ICML 2017. arXiv:1703.04730
- Pruthi, Liu, Kale, Sundararajan. *Estimating Training Data Influence by Tracing Gradient Descent (TracIn).* NeurIPS 2020. arXiv:2002.08484
- Basu, Pope, Feizi. *Influence Functions in Deep Learning Are Fragile.* ICLR 2021. arXiv:2006.14651
- Bae, Ng, Lo, Ghassemi, Grosse. *If Influence Functions are the Answer, Then What is the Question?* NeurIPS 2022. arXiv:2209.05364
- Ilyas, Park, Engstrom, Leclerc, Madry. *Datamodels: Predicting Predictions from Training Data.* ICML 2022. arXiv:2202.00622
- Park, Georgiev, Ilyas, Leclerc, Madry. *TRAK: Attributing Model Behavior at Scale.* ICML 2023. arXiv:2303.14186
- Guu, Webson, Pavlick, Dixon, Tenney, Bolukbasi. *Simfluence: Modeling the Influence of Individual Training Examples by Simulating Training Runs.* 2023. arXiv:2303.08114 †

**Compositional generalization and formal languages**
- Lake, Baroni. *Generalization without Systematicity (SCAN).* ICML 2018. arXiv:1711.00350
- Kim, Linzen. *COGS: A Compositional Generalization Challenge Based on Semantic Interpretation.* EMNLP 2020. arXiv:2010.05465
- Keysers et al. *Measuring Compositional Generalization: A Comprehensive Method on Realistic Data (CFQ).* ICLR 2020. arXiv:1912.09713
- Hupkes, Dankers, Mul, Bruni. *Compositionality Decomposed: How do Neural Networks Generalise?* JAIR 2020. arXiv:1908.08351
- Csordás, Irie, Schmidhuber. *The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers.* EMNLP 2021. arXiv:2108.12284
- Hahn. *Theoretical Limitations of Self-Attention in Neural Sequence Models.* TACL 2020. arXiv:1906.06755
- Delétang et al. *Neural Networks and the Chomsky Hierarchy.* ICLR 2023. arXiv:2207.02098

**Representations and causal interpretability**
- Kornblith, Norouzi, Lee, Hinton. *Similarity of Neural Network Representations Revisited (CKA).* ICML 2019. arXiv:1905.00414
- Song, Smola, Gretton, Bedo, Borgwardt. *Feature Selection via Dependence Maximization.* JMLR 13, 2012. (Unbiased HSIC.)
- Ding, Denain, Steinhardt. *Grounding Representation Similarity with Statistical Testing.* NeurIPS 2021. arXiv:2108.01661
- Davari, Horoi, Natik, Lajoie, Wolf, Belilovsky. *Reliability of CKA as a Similarity Measure in Deep Learning.* ICLR 2023. arXiv:2210.16156 †
- Geiger, Lu, Icard, Potts. *Causal Abstractions of Neural Networks.* NeurIPS 2021. arXiv:2106.02997

**Architecture and optimization**
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762
- Radford et al. *Language Models are Unsupervised Multitask Learners (GPT-2).* 2019.
- Xiong et al. *On Layer Normalization in the Transformer Architecture.* ICML 2020. arXiv:2002.04745
- Loshchilov, Hutter. *Decoupled Weight Decay Regularization (AdamW).* ICLR 2019. arXiv:1711.05101

**Statistics and experimental methodology**
- Schuirmann. *A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability.* J. Pharmacokinetics and Biopharmaceutics 15(6), 1987.
- Lakens. *Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses.* Social Psychological and Personality Science 8(4), 2017.
- Holm. *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian Journal of Statistics 6(2), 1979.
- Efron, Tibshirani. *An Introduction to the Bootstrap.* Chapman & Hall, 1993.
- Bouthillier et al. *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021. arXiv:2103.03098
- Agarwal, Schwarzer, Castro, Courville, Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS 2021. arXiv:2108.13264
- Zhuang, Zhang, Sculley et al. *Randomness in Neural Network Training: Characterizing the Impact of Tooling.* MLSys 2022. arXiv:2106.11872

**Technical documentation**
- PyTorch, *Reproducibility*: https://pytorch.org/docs/stable/notes/randomness.html
- PyTorch, `torch.use_deterministic_algorithms`: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
- NVIDIA cuBLAS, *Results reproducibility*: https://docs.nvidia.com/cuda/cublas/#results-reproducibility
- NumPy, *Parallel random number generation (SeedSequence)*: https://numpy.org/doc/stable/reference/random/parallel.html
- safetensors: https://github.com/huggingface/safetensors
- SciPy, `scipy.stats.bootstrap`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- statsmodels, `ttost_paired`: https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttost_paired.html
- statsmodels, `multipletests`: https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
