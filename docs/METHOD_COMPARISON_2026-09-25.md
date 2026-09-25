# MSA-BP vs. Pixel, Polygon and Diffusion Antenna Representations

*Qualitative comparison, 2026-09-25. Entries describe typical formulations of each family; no head-to-head benchmark has been run yet.*

**MSA-BP (ours)** builds the antenna from engineering primitives: main slot, stepped patch bricks, corner notches and ears, CPW feed with matching stubs, and slot branches. Each primitive is sized by ratio parameters K ∈ [0, 1], scaled against caps measured on the current metal patch by ray/strip casting. Every point of the unit box therefore maps to a valid, mirror-symmetric layout.

| | Pixel-based | Polygon-based | Diffusion-based | **MSA-BP (ours)** |
|---|---|---|---|---|
| **Representation** | Binary material grid | Free vertex / control-point coordinates | Generative model over layout images | Named primitives sized by ratio parameters |
| **Search dimension** | Very high (10²–10³+ bits) | Medium (2 × vertices) | High latent space, fixed after training | **Low (23 in the current model)** |
| **Validity of a random sample** | Needs repair: corner-only contacts, floating islands | Self-intersections and vertex-order violations | Not guaranteed; needs thresholding and vectorization | **Valid by construction** |
| **Chance of stair / notch / stub structures** | Only at pixel granularity, by chance | Very low: needs several vertices aligned at right angles at once | Only if the training set contains them | **Always present; the optimizer only tunes their size** |
| **Topological freedom** | Highest (within grid resolution) | Low: genus fixed, no new slots or holes | High, but bounded by the training distribution | Fixed skeleton; the branch-tree extension adds or removes branches |
| **Geometry fidelity / CAD export** | Staircase edges, resolution-limited | Exact vectors | Raster output, lossy vectorization | **Exact polygons, direct CST import** |
| **Fit for sample-efficient optimization (BO, surrogate-assisted EA)** | Poor: binary and high-dimensional, so GA/BPSO with many EM runs | Moderate: infeasible regions hurt surrogates | Indirect: guidance or conditioning, EM verification still needed | **Good: low-dimensional continuous unit box (K-RVEA, qLogEHVI)** |
| **Up-front cost** | None | None | Large simulated dataset plus training | Designer effort to define the primitives |
| **Interpretability** | Low | Low–medium | Low | **High: each variable is a position, width or length of a named feature** |
| **Main weakness** | EM cost, manufacturability | Infeasibility, feature-poor shapes | Data cost, no validity or performance guarantee | Expressiveness limited to the primitive vocabulary |

## Why polygon methods rarely grow stairs, notches or stubs

Each of these features is a coordinated pattern: several vertices aligned on a common axis, meeting at right angles, often around a narrow gap. When vertices are perturbed independently, the chance of landing on such a pattern is vanishingly small, so neither random search nor a surrogate model will find one in practice. In MSA-BP these features are primitives: they exist in every sample, and the search only decides how big they are and where they sit.

## Where MSA-BP is weaker

- **Bounded expressiveness.** Features are axis-aligned rectangles on a designer-chosen skeleton. A genuinely new topology is out of reach unless it is added as a primitive or grown through the branch tree.
- **Variable dimension of the branch tree.** Each added branch brings three more K parameters, so the tree cannot yet feed fixed-dimension optimizers directly.
- **No absolute minimum feature size.** Only relative lower bounds (K ≥ 0.05) and fixed clearances are enforced so far.
