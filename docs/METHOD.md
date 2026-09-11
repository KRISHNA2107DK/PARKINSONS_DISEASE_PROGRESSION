# Method — S-PBGL derivation

The core methodological contribution of BrainFormer-PD v2.1. This document
reproduces the mathematical derivation in detail for reviewer verification.

## Symbol table

| Symbol | Shape | Meaning |
|---|---|---|
| `z_i` | [B, d_model] | Patient embedding from multi-modal encoder + LTP |
| `μ_F^(i)` | [B, n_nodes, d_node] | Learned posterior mean over node features (Tanh activation) |
| `log σ_F^(i)` | [B, n_nodes, d_node] | Learned posterior log-variance (clamped to [-4, 2]) |
| `σ_F^(i)` | [B, n_nodes, d_node] | `exp(0.5 * log σ)` |
| `node_miss_i` | [B, n_nodes] | Per-node missingness fraction from modality missingness × NODE_WEIGHTS |
| `κ` | scalar | Learned missingness inflation parameter (init 0.5) |
| `F_i` | [B, n_nodes, d_node] | Sampled node features: `μ + σ ⊙ ε` (training) or `μ` (inference) |
| `S_i` | [B, n_nodes, n_nodes] | Bilinear affinity: `F_i F_iᵀ / √d_node` |
| `A_braak` | [n_nodes, n_nodes] | Canonical Braak 10-region prior adjacency |
| `ε₀` | scalar | `braak_eps`, small offset so all edges are possible (0.05) |
| `s` | scalar | Learned temperature (`scale`, init 0.1) |
| `A_i` | [B, n_nodes, n_nodes] | Posterior mean personalised adjacency, symmetric-normalised |
| `u_edge^(i)` | [B, n_nodes, n_nodes] | Per-edge epistemic uncertainty (variance) |
| KL | scalar | KL divergence of node-feature posterior against N(0, I) |

## Step 1 — Posterior over node features

Two separate MLPs project the patient embedding to a Gaussian posterior over the 8-dimensional
feature of each of 10 Braak nodes:

```
μ_F^(i)     = Tanh(z_i W_μ + b_μ)            reshape → (n_nodes, d_node)
log σ_F^(i) = z_i W_σ + b_σ                  reshape → (n_nodes, d_node), clamp[-4, 2]
σ_F^(i)     = exp(½ log σ_F^(i))
```

## Step 2 — Missingness-induced variance inflation

```
σ_F^(i)[j] ← σ_F^(i)[j] · exp(κ · node_miss_i[j])
```

Multiplicative inflation. When a modality is absent, the node's uncertainty grows — rather than
the node's value defaulting to zero. `κ` is learned. This replaces the earlier heuristic
"zero out edges of missing nodes" approach (which was biologically implausible: the pathway
is not absent, it is simply unobserved).

## Step 3 — Reparameterisation vs. inference

- **Training**: `F_i = μ_F + σ_F ⊙ ε`, `ε ~ N(0, I)` (standard VAE reparameterisation).
- **Inference**: `F_i = μ_F` — use the posterior mean, no stochastic sampling.

## Step 4 — Posterior mean adjacency with symmetric normalisation

```
S_i      = F_i F_iᵀ / √d_node                          # bilinear affinity
A_prior  = A_braak + ε₀                                # allow any edge with small floor
A_raw    = σ(s · S_i) ⊙ A_prior                        # sigmoidal, Braak-masked
D_i      = diag(∑_k A_raw[:, k])                       # row-sum degrees
A_i      = D_i⁻½ · A_raw · D_i⁻½                      # symmetric normalisation
```

The symmetric normalisation is essential: it matches the spectral-GCN formulation and ensures
A_i has eigenvalues bounded in [−1, 1] for stable graph convolution downstream.

## Step 5 — Analytical edge uncertainty (the key contribution)

`S_i[j, k]` is the dot product of two **independent** Gaussian vectors `F_i[j]` and `F_i[k]`:

```
F_i[j] = μ_F[j] + σ_F[j] ⊙ ε_j      ε_j ~ N(0, I_d)
F_i[k] = μ_F[k] + σ_F[k] ⊙ ε_k      ε_k ~ N(0, I_d)
```

Its mean is `μ_F[j] · μ_F[k]ᵀ / √d_node`. For the variance, apply the identity for
the variance of a sum of products of independent Gaussian random variables (which reduces to
the formula below because all cross-terms vanish in expectation):

```
Var[S_i[j, k]] = (||σ_F[j]||² · ||μ_F[k]||²
               + ||μ_F[j]||²  · ||σ_F[k]||²
               + ||σ_F[j]||² · ||σ_F[k]||²) / d_node
```

This is **exact** — no Monte-Carlo sampling needed. All three terms are computable from
`μ_F` and `σ_F` at O(n_nodes² · d_node) per patient per batch.

Propagate through the sigmoid via the delta method, scaled by the Braak prior:

```
p̄ = σ(s · S_mean) ⊙ A_prior                                  # mean edge probability
u_edge[j, k] = s² · p̄[j, k]² · (1 − p̄[j, k])² · Var[S[j, k]] · (A_prior[j, k])²
```

The `p̄² (1−p̄)²` factor is the squared derivative of the sigmoid; the `A_prior²` factor
propagates the Hadamard masking into the variance.

The result is a per-patient, per-edge uncertainty map that is interpretable: large `u_edge`
means the model cannot confidently assert whether the edge is "active" in that patient's
Braak-propagation topology.

## Step 6 — KL regularisation

Per the standard VAE derivation, the KL divergence of the posterior from a standard normal
prior `N(0, I)` is:

```
KL = ½ · Σ_j Σ_d (μ_F[j, d]² + σ_F[j, d]² − 1 − log(σ_F[j, d]²))
```

Summed over nodes and feature dimensions, averaged over the batch. Annealed into the total
loss with weight `λ_KL = min(epoch/30, 1) · 10⁻⁴`. The prior is `N(0, I)` rather than a
Braak-structured prior because the Braak structure is already encoded in the likelihood via
`A_prior` masking in Step 4 — adding it to the prior would over-constrain the model.

## Step 7 — How A_i is consumed downstream

`A_i` is used in two places:

1. **Graph convolution (GMP-PBG)**: 2-round message passing over the imaging feature vector
   projected onto the 10 Braak nodes. Injects structural context into the patient embedding.
2. **Position-sensitive temporal attention (BSMTA)**: per-head temporal-node queries propagate
   through `A_i` to produce T×T attention bias. Gives the transformer head a biological
   prior over which visit-pairs should attend.

The S-PBGL posterior variance `u_edge` is exported as an interpretability output — it does
NOT enter the downstream computation. This is deliberate: `u_edge` is a diagnostic, not a
regularisation signal.

## Computational complexity

- S-PBGL forward pass: O(B · n_nodes² · d_node) per batch.
- Analytical variance: same O(B · n_nodes² · d_node) — no sampling.
- Memory: 3 extra matrices of shape `[B, n_nodes, n_nodes]` (A_i, u_edge, A_raw) vs. point-estimate PSBGL.
- Wall-clock overhead on RTX 5060 vs. deterministic PSBGL: approximately 12% (measured across batches of 64).

## Why not MC-dropout or variational inference over weights?

- MC-dropout requires K forward passes per prediction at inference, quadrupling latency.
- BayesianGCN-style weight posteriors entangle model uncertainty with data uncertainty and are hard to interpret clinically.
- The bilinear-form variance identity gives an **analytical**, **per-edge**, **interpretable**
  uncertainty at zero sampling cost. This is the specific technical advantage of S-PBGL over
  prior stochastic-graph frameworks (VGAE, NRI, DropEdge, BayesianGCN).

## References

- Kingma & Welling (2014) — the standard VAE reparameterisation.
- Amini et al. (NeurIPS 2020) — evidential regression with NIG prior (used for the outcome head, not the graph).
- Kipf & Welling (2017) — symmetric-normalised graph convolution.
- Khosla et al. (2020) — supervised contrastive learning.
- Braak & Braak (2003) — the canonical 6-stage α-synuclein propagation model underlying `A_braak`.
