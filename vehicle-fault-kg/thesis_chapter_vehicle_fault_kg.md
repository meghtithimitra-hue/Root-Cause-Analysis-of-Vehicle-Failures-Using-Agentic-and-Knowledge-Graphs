# Vehicle-Fault-KG: A Knowledge-Graph-Driven Reasoning Pipeline for Automotive Fault Diagnosis

> *This chapter presents Vehicle-Fault-KG, an early version of the diagnostic pipeline developed within the present thesis. It is concerned exclusively with knowledge-graph grounding of the retrieval and reasoning stages; subsequent pipeline versions extend the architecture with time-series sensor evidence and simplify the decision space to two operational modes. The chapter documents the design rationale, the constituent algorithms, the empirical characterisation of the constructed artefact, and the architectural decisions that shaped later iterations.*

---

## 4.1 Introduction

Automotive fault diagnosis is a knowledge-intensive activity. Practitioners reason over a hierarchical decomposition of the vehicle into systems, subsystems, symptoms, and diagnostic procedures, and they do so under uncertainty: a single observed symptom is compatible with several candidate faults, and the discriminating evidence is often obtained only through a sequence of physical inspections. The arrival of large language models (LLMs) has made free-form diagnostic dialogue technically feasible, yet the ungrounded generation of these models poses a fundamental risk in a domain where an erroneous conclusion carries material consequence.

The central proposition of the present thesis is that diagnostic reasoning should be *grounded in a structured representation of domain knowledge* rather than in parametric memory alone. This chapter describes Vehicle-Fault-KG, a complete, self-contained implementation of that proposition: a pipeline that constructs a hierarchical knowledge graph from a curated automotive fault corpus, retrieves diagnostically relevant knowledge through a hybrid of semantic, structural, and community-level evidence, and produces an answer whose provenance is traceable to explicit graph nodes and edges.

Vehicle-Fault-KG constitutes a *version of the pipeline* in the sense that it fixes the architectural skeleton — staged graph construction, multi-path retrieval, evidence-based confidence scoring, and mode-based answer generation — that later versions retain while replacing or augmenting individual stages. Its principal contribution is a disciplined treatment of the retrieval-to-decision interface: a candidate scoring scheme that separates the *ranking* of candidates from the *decision* to commit to an answer, together with an explicit three-mode operational semantics (*extracted*, *inferred*, *ambiguous*) that governs whether the system asserts a diagnosis, proposes a partial hypothesis for refinement, or declines to answer.

The remainder of the chapter is organised as follows. Section 4.2 positions the work within the relevant literature and establishes the theoretical foundations on which the pipeline rests. Section 4.3 presents the overall architecture. Section 4.4 describes the construction of the knowledge graph and its empirical characterisation. Section 4.5 formalises the hybrid retrieval layer. Section 4.6 develops the confidence and decision model. Section 4.7 describes reasoning-path construction and answer generation. Section 4.8 documents the implementation, and Section 4.9 evaluates the artefact and discusses its limitations. Section 4.10 summarises the chapter and its implications for later pipeline versions.

## 4.2 Related Work and Theoretical Foundations

### 4.2.1 Knowledge graphs in diagnostic reasoning

The use of graph-structured domain knowledge in fault diagnosis has a long tradition, from decision trees and fault trees to Bayesian networks over causal graphs. Knowledge graphs offer a more flexible substrate: they encode entities and typed relations without committing to a fixed inference topology, and they support both deductive traversal (walking from a system to its subsystems and symptoms) and inductive inference (connecting entities that co-occur across sources). In the automotive domain, fault knowledge is naturally hierarchical — a *category* (e.g., ABS System) contains *subcategories* (e.g., ABS Control Module), each associated with *symptoms* and ordered *diagnostic procedures*. This study adopts that native structure as the backbone of the graph.

A recurring problem in automatically constructed knowledge graphs is the *confidence* attached to edges. The present work distinguishes two provenance classes, consistent with the distinction in the information-extraction literature between *extracted* facts, which are explicitly stated in a source, and *inferred* facts, which are derived by a rule and are therefore subject to a residual risk of error. Vehicle-Fault-KG preserves this provenance on every edge, allowing the downstream confidence model to weight evidence accordingly.

### 4.2.2 Hybrid retrieval

Retrieval over a knowledge graph is a multi-criteria problem. Pure lexical matching is brittle in the face of paraphrase ("spongy brake pedal" vs. "soft brake pedal"); pure semantic (embedding-based) search is insensitive to the graph's explicit hierarchical structure; and graph traversal alone cannot generalise to vocabulary absent from node labels. This motivates a *hybrid* design in which independent retrieval signals are fused. The design principle adopted here is that of *agreement*: candidates corroborated by multiple, mutually independent retrieval paths are more likely to be genuinely relevant than candidates surfaced by any single path, and should therefore receive a monotonic boost in score. This is a retrieval analogue of the ensemble principle in machine learning, and it forms the quantitative core of the fusion strategy described in Section 4.5.

### 4.2.3 Retrieval-augmented generation and GraphRAG

Retrieval-augmented generation (RAG) mitigates the hallucination tendency of generative models by conditioning generation on externally retrieved evidence. GraphRAG extends the paradigm by organising the retrieved evidence as a graph, which enables the answer to cite an explicit reasoning path rather than a flat list of passages. The architecture of Vehicle-Fault-KG follows the GraphRAG template: retrieval first, then *graph-side* reasoning, then generation. A deliberate design decision is that the graph itself must be able to answer a substantial class of queries *without* invoking a generative model at all; the LLM is reserved for cases in which the graph evidence is insufficient, and its output is ultimately overridden by a conservative no-match response. This ordering reflects the thesis' position that generative models should assist, but never solely determine, diagnostic conclusions.

### 4.2.4 Uncertainty and probabilistic evidence fusion

The confidence model rests on two classical notions. The first is *probabilistic evidence fusion* in the form of the noisy-OR combination rule, which models the conjunctive support of independent evidence sources: for independent sources with support $x$ and $y$, the combined support is $1 - (1 - x)(1 - y)$. The second is the *saturating* transformation, a concave mapping that yields the greatest relative gain for mid-range evidence and asymptotically diminishing gains as evidence approaches certainty. Both devices are used to translate raw retrieval scores — which are not calibrated probabilities — into quantities that behave more like evidential strengths. The calibration stage, described in Section 4.6.1, is explicitly framed as a score-to-evidence mapping, not as a claim of calibrated posterior probability.

## 4.3 System Architecture

### 4.3.1 Overview

Vehicle-Fault-KG is organised into a strictly staged architecture in which the output of each stage is the input to the next, and no stage revisits the decisions of its predecessors. The pipeline is separable into an *offline* construction phase and an *online* reasoning phase.

The offline phase, executed once, transforms a curated JSON corpus of automotive fault knowledge into three persistent artefacts: (i) a hierarchical, community-labelled knowledge graph; (ii) a vector index over graph nodes; and (iii) a community membership map. The online phase consumes a natural-language fault description and produces a mode-labelled answer by chaining five stages:

1. **Hybrid retrieval** — a semantic, structural, and community-aware candidate search;
2. **Score calibration** — a monotone, source-aware rescaling of candidate scores;
3. **Confidence scoring** — the computation of a per-candidate effective confidence and an overall decision mode;
4. **Reasoning-path construction** — a traceable walk of the graph from matched nodes to their systems and subsystems, with symptom confirmation;
5. **Answer generation** — a mode-dependent dispatch between a graph-sourced answer, an intermediate refinement prompt, a clarifying question, or a no-match response.

Figure 4.1 depicts the end-to-end data flow.

```
 Offline construction                           Online reasoning
 ──────────────────                            ─────────────────
 Curated corpus ──► Extraction ──► Graph ──► Hybrid Retrieval ──► Calibration
                      │              │        (vector + graph      │
                      │              │         + community)        ▼
                      ▼              ▼                          Confidence
                 Normalisation   Communities ────────────────►   Scoring
                      │              │                          │
                      ▼              ▼                          ▼
                 Vector index   Community map             Reasoning Path
                                                                 │
                                                                 ▼
                                                           Answer Generation
                                                                 │
                                        ┌────────────────────────┼────────────────┐
                                        ▼                        ▼                ▼
                                   EXTRACTED               INFERRED            AMBIGUOUS
                                  full diagnosis    intermediate refine     clarify / no-match
```

**Figure 4.1 — End-to-end architecture of Vehicle-Fault-KG.** Offline artefacts (left) feed the five-stage online pipeline (right). The decision stage partitions the answer space into three operational modes.

### 4.3.2 The corpus

The knowledge substrate is a curated corpus of 99 records, each of the form $(\text{category}, \text{subcategory}, \text{symptoms}, \text{diagnosis\_steps})$, where a symptom is a natural-language phrase and a diagnosis step is a phrase paired with a two-element result pair (e.g., a step "Check ABS fuse" with results "Blown"/"Intact"). The corpus spans 13 top-level systems and 98 subsystems, providing full coverage of the conventional automotive fault taxonomy used in the thesis. Its provenance is a structured diagnostic reference, pre-formatted into the record schema; no free-text extraction from unstructured prose is required at this version, a simplification revisited in Section 4.9.4.

## 4.4 Knowledge Graph Construction

### 4.4.1 Entity extraction and canonicalisation

Each record is decomposed into four entity types: `Category` (13), `Subcategory` (98), `Symptom` (143), and `DiagnosisStep` (190). Entities are canonicalised by a content-addressed identifier scheme,

$$
\mathrm{id} = \text{prefix}\_\;\mathrm{slug}(\mathrm{label})_{[0:40]}\_\;\mathrm{md5}(\mathrm{label})_{[0:8]},
$$

where $\mathrm{slug}$ lowercases the label and replaces every non-alphanumeric run with a single underscore. Because the identifier is a deterministic function of the label, identical phrases expressed in different records resolve to the same node, which provides a first line of de-duplication at the *extraction* level, prior to any algorithmic deduplication.

Two intra-record mappings are recorded at extraction time for provenance: the first source file in which a given symptom or step appears, and the *set* of categories in which it occurs. The latter becomes critical for the generation of cross-category edges (Section 4.4.3).

### 4.4.2 Node and edge schema

Every node carries the fields `id`, `label`, and a `source_file` pointer into the corpus. Diagnosis-step nodes additionally carry the `result_a` and `result_b` outcomes. A subsequent normalisation pass enriches every node with `node_type`, `category`, and `subcategory` attributes — resolved by label lookup against the raw corpus and, failing that, by parsing the `source_file` path — so that downstream stages can operate on a uniform, self-describing schema without re-consulting the corpus.

The edge schema distinguishes hierarchical edges, which are *extracted* directly from the record structure, from similarity edges, which are *inferred*:

- `HAS_SUBCATEGORY` — category to subcategory (extracted, weight 1.0);
- `HAS_SYMPTOM` — subcategory to symptom (extracted, weight 1.0);
- `HAS_DIAGNOSIS_STEP` — subcategory to diagnosis step (extracted, weight 1.0);
- `SIMILAR_SYMPTOM_TO` — inferred between subcategories of *different* categories that share a symptom (weight 0.8);
- `SIMILAR_STEP_TO` — inferred between subcategories of *different* categories that share a diagnosis step (weight 0.8).

The inferred edges are generated by a single rule: whenever a symptom (or step) occurs under subcategories belonging to more than one category, all cross-category pairs of those subcategories are connected, and the connecting edge is labelled with the shared symptom (or step). These edges are the graph's mechanism for *hypothesis generation*: they encode the empirical observation that a symptom bridges multiple systems and hence that a diagnostic hypothesis cannot be confined to a single category. Their reduced weight (0.8) and their `INFERRED` provenance label encode the designer's prior that derived connections are less certain than extracted ones. Table 4.1 summarises the final edge distribution.

**Table 4.1 — Edge distribution of the constructed graph (444 nodes, 562 edges).**

| Relation              | Count | Provenance |
|-----------------------|------:|------------|
| `HAS_DIAGNOSIS_STEP`  |   196 | Extracted  |
| `HAS_SYMPTOM`         |   194 | Extracted  |
| `HAS_SUBCATEGORY`     |    99 | Extracted  |
| `SIMILAR_SYMPTOM_TO`  |    71 | Inferred   |
| `SIMILAR_STEP_TO`     |     2 | Inferred   |

Of the 562 edges, 203 connect nodes belonging to different systems — the cross-category connectivity that the similarity rule deliberately introduces.

### 4.4.3 Deduplication and community detection

The extraction-level identifier scheme already guarantees label-unique nodes; a graphify-based de-duplication pass is nevertheless applied to absorb residual near-duplicate entities. In the current corpus the pass is effectively a no-op on node counts, but it is retained as part of the construction contract for robustness against future, less-clean corpora. Single-threaded execution is forced at this stage to avoid numerical-layout races in the underlying BLAS libraries on the Windows platform.

Community structure is induced by Leiden clustering of the directed graph as provided by the construction toolkit, with a seeded Louvain fallback (seed 42) should the primary implementation be unavailable. The resulting partition assigns each node a `community` attribute and drives the community search path of the retrieval layer (Section 4.5.3). The partition yields 17 communities with a strongly heterogeneous size distribution: 13 communities (76.5%) are *multi-category*, i.e., they contain nodes from two or more systems, and the largest communities correspond precisely to the most cross-connected systems. The four single-category communities are small (5 nodes each) and are confined to engine components, reflecting the comparatively low internal connectivity of that portion of the graph. This empirical structure — a majority of cross-cutting communities — is a direct consequence of the inferred similarity edges and is itself evidence that cross-system fault relationships dominate the knowledge substrate.

### 4.4.4 Analytic characterisation

The construction stage concludes with a structural analysis of the graph. The highest-degree nodes — the *god nodes* — are dominated by engine-system entities: Fuel Injector (degree 17), Engine Components (15), Electrical System (14), Engine Control Unit (14), and Ignition Wire Set (13). Their centrality reflects both the granularity of the engine taxonomy in the corpus and the concentration of shared symptoms around engine subsystems. The most *surprising* connections — pairs whose linkage is not apparent from a single source document — are dominated by the inferred similarity edges (e.g., Brake Hose ↔ Brake Fluid, Coolant/Antifreeze ↔ Coolant Reservoir), which is the intended behaviour of the rule: it surfaces latent relationships that a document-by-document reading would not expose.

## 4.5 Hybrid Retrieval

Given a natural-language query $q$, the retrieval layer returns an ordered candidate list $\mathcal{C}$ drawn from the graph. Retrieval proceeds through three independent paths, each of which is a separate estimation of relevance, followed by a fusion step that rewards agreement.

### 4.5.1 Vector path

The vector path embeds every indexable node — Subcategory, Symptom, and DiagnosisStep nodes; Category nodes are deliberately excluded as overly coarse for retrieval — into a shared dense space using the `all-MiniLM-L6-v2` sentence transformer. Each node's embedding is computed over a *contextualised* text that prefixes the node's position in the taxonomy (e.g., a symptom is embedded as `System > Subsystem | Symptom: <label>`, and a diagnosis step additionally appends its result pair). The query is embedded with the same model, and cosine similarity against the ChromaDB index yields the raw score

$$
s_v(n) = 1 - d_{\cos}(q, n),
$$

with the $k=10$ nearest nodes returned. This path captures semantic equivalence across phrasing differences.

### 4.5.2 Structural path

The structural path operates directly on the graph. Query terms of length greater than two characters seed the traversal by substring matching against node labels; from each seed the graph is expanded by breadth-first search up to a horizon of $h_{\max} = 2$ hops along outgoing edges. A candidate $n$ at hop distance $h(n)$ with label-word overlap $w(n)$ against the query receives

$$
s_g(n) = \max\bigl(0,\; 1 - 0.3\,h(n) + 0.1\,w(n)\bigr),
$$

so that direct label matches dominate, one-hop context (parent categories, sibling subcategories) is strongly rewarded, and two-hop context decays to a modest baseline. Up to 30 candidates are retained. This path captures the hierarchical context that flat semantic search cannot.

### 4.5.3 Community path

The community path operationalises the graph's community structure as a retrieval signal. The top three vector-path results cast a majority vote over their community memberships; the winning community is selected, and *every* node of that community becomes a candidate with a uniform base score

$$
\beta(c) = \begin{cases} 0.6, & \text{if } c \text{ is multi-category},\\ 0.4, & \text{otherwise,} \end{cases}
$$

capped at $k = 10$ candidates. The premium on multi-category communities encodes the design position that communities spanning multiple systems represent genuine, higher-value cross-system fault relationships and should therefore outrank single-system clusters when acting as the sole source of evidence.

### 4.5.4 Fusion

The three candidate sets are merged by node identity. A node's final score is its vector score where available, a flat structural baseline (0.3) where only the graph found it, or the community base score where only the community found it; crucially, every *additional* independent path that corroborates a node adds a fixed positive boost. Table 4.2 states the complete fusion policy.

**Table 4.2 — Fusion policy: score contributions by evidence combination.**

| Evidence combination        | Score                                                            |
|-----------------------------|------------------------------------------------------------------|
| vector only                 | $s_v$                                                            |
| graph only                  | 0.30                                                             |
| community only              | 0.40 (0.60 multi-category)                                       |
| vector + graph              | $s_v + 0.5$                                                      |
| vector + community          | $s_v + 0.3$                                                      |
| graph + community           | 0.50                                                             |
| all three                   | $s_v + 1.3$                                                      |

Two remarks are in order. First, the boosts are *additive and independent of magnitude*; they are intended as an ordinal reward for agreement rather than a calibrated magnitude. Second, an inconsistency between the implemented fusion and its accompanying specification is documented in the source (the documented "base + 0.8" for the all-three case describes only the community-step increment, whereas the cumulative boost applied by the merge loop is $0.5 + 0.8 = 1.3$). The downstream calibration stage must therefore invert the *implemented* offsets, and the present chapter follows the implementation. This discrepancy is a concrete instance of the specification-drift hazard discussed in Section 4.9.4.

The output contract of the retrieval stage — each candidate carrying `node_id`, `node_type`, `label`, `category`, `subcategory`, `score`, `source` (the evidence combination), `community_id`, and `is_multi_category` — is the interface consumed by every subsequent stage.

## 4.6 Confidence Modelling and the Decision Stage

The retrieval layer produces a *ranking*; it does not, by itself, license a decision. The decision stage introduces an explicit, threshold-based semantics, decomposing the task into a calibration mapping and a path-aware confidence composition.

### 4.6.1 Score calibration

Raw retrieval scores occupy heterogeneous ranges: a pure vector score lives in $[0,1]$, a graph-only score is a flat 0.3, a community-only score is 0.4 or 0.6, and boosted combinations can exceed 1.0 (up to $\approx 1.3$). Calibration first *decomposes* a combined score into its vector component by subtracting the source-specific offset (Table 4.2), then clamps negative values to zero, applies a saturating transformation with coefficient $\alpha = 0.4$,

$$
\hat{s} = s \cdot \bigl(1 + \alpha\,(1 - s)\bigr),
$$

and finally fuses the vector evidence with a structural baseline by the noisy-OR rule,

$$
c = 1 - (1 - \hat{s})\,(1 - \beta),
$$

where $\beta$ is the graph baseline (0.3), the community baseline (0.4), or their noisy-OR combination ($1 - 0.7\cdot 0.6 = 0.58$) depending on which structural paths corroborate the candidate. Pure vector candidates are clamped only; pure structural candidates pass through unchanged, as they contain no vector component to decompose. Calibration is strictly monotone in the raw score for a fixed source, so it cannot change the retrieval ranking within a source class; its purpose is to make scores *comparable across* source classes and to compress the unreasonably large boosted magnitudes into an evidential range.

### 4.6.2 Path-aware effective confidence

The calibrated score still overstates confidence in one respect: the evidence sources are not fully independent. The graph path and the community path are *both* derived from the same underlying graph, so their agreement is partially correlated; and a single-path result carries no corroboration at all. The confidence model therefore multiplies the calibrated score by a *path-confidence factor* $\rho(\text{source})$ that discounts correlated or single-path evidence. Table 4.3 states the factors.

**Table 4.3 — Path-confidence factors $\rho$ by evidence source.**

| Source              | $\rho$ | Rationale                                        |
|---------------------|-------:|--------------------------------------------------|
| all three paths     |   1.00 | Three mutually complementary signals             |
| vector + graph      |   0.95 | Semantic and structural — near-orthogonal        |
| vector + community  |   0.85 | Community partially derived from the graph       |
| graph + community   |   0.70 | Both structural — correlated, no semantic signal |
| vector only         |   0.55 | Single semantic path                             |
| community only      |   0.45 | Single, partially derived path                   |
| graph only          |   0.35 | Single structural path                           |

A multi-category community membership confers an additional bonus of 0.10 for community-sourced candidates. The effective confidence of candidate $n$ is therefore

$$
C(n) = s_{\mathrm{cal}}(n) \cdot \rho\bigl(\mathrm{source}(n)\bigr).
$$

### 4.6.3 Operational modes

The overall decision mode is taken from the top-ranked candidate, and each candidate is tagged with a mode according to two thresholds on its effective confidence:

$$
\mathrm{mode}(n) = \begin{cases} \text{EXTRACTED}, & C(n) \ge 0.75,\\ \text{INFERRED}, & 0.55 \le C(n) < 0.75,\\ \text{AMBIGUOUS}, & C(n) < 0.55. \end{cases}
$$

The semantics are deliberate. **EXTRACTED** asserts that the top candidate is directly corroborated by multiple independent retrieval paths and permits the system to commit to a full diagnosis. **INFERRED** signals a partial match: the system can identify a likely system and subsystem but cannot confirm the fault on the evidence presented, and it responds with an intermediate, refinement-seeking answer rather than a diagnosis. **AMBIGUOUS** is the refusal state: the evidence does not support any commitment, and the system issues a clarifying question or a no-match response. The three-mode structure — in particular the intermediate state — is the signature design decision of this pipeline version, and its compression to two modes in later versions is analysed in Section 4.10.

### 4.6.4 Clarification

When the mode is AMBIGUOUS, the system constructs a clarifying question from the top candidate labels, requesting the discriminating details that the graph most lacks (symptom type, occurrence conditions, warning lights, recent repairs). This is a *guided*, not generic, prompt: it is grounded in the retrieved evidence and names the specific entities that were matched, so that the follow-up dialogue steers the query toward evidence the graph can adjudicate. The interface permits the user either to answer (re-entering the pipeline with the enriched query) or to skip clarification and force a final response (Section 4.7.3).

## 4.7 Reasoning and Answer Generation

### 4.7.1 Reasoning-path construction

For the top five EXTRACTED or INFERRED candidates, the reasoner reconstructs the candidate's position in the graph by walking *up* the hierarchical edges to its Subcategory and Category parents, and *across* to the siblings of its subcategory. Two structural products emerge:

- **Symptom evidence.** The sibling symptoms of the matched subcategory are partitioned into *matched symptoms* — those whose labels share query terms (words longer than two characters) — and *unconfirmed symptoms*, which are present in the graph for the subsystem but not evidenced by the query. This partition is the quantitative basis of the INFERRED-intermediate response.
- **Diagnostic procedures.** The diagnosis steps attached to the matched subcategory are collected as the recommended inspection sequence.

The walk is rendered as a human-readable reasoning chain of the form *"Symptom S matched — under Subcategory X; Subcategory X belongs to Category Y; related diagnosis steps …"*, providing full traceability from the final answer back to explicit graph nodes.

### 4.7.2 Mode-based answer dispatch

The answer generator is a pure function of the mode and the reasoning path (Table 4.4).

**Table 4.4 — Answer dispatch policy.**

| Mode        | Condition                             | Response                                                        | Evidence source |
|-------------|---------------------------------------|-----------------------------------------------------------------|-----------------|
| EXTRACTED   | —                                     | Full diagnosis: system, subsystem, matched symptoms, steps, chain | graph           |
| INFERRED    | Unconfirmed symptoms exist            | Intermediate summary (system, subsystem) + refinement prompts    | graph           |
| INFERRED    | No unconfirmed symptoms               | Full diagnosis, as EXTRACTED                                     | graph           |
| AMBIGUOUS   | Clarification skipped (`skip`)        | Fixed no-match directive                                          | no_match        |
| AMBIGUOUS   | Clarification not skipped             | Guided clarifying question                                        | graph           |

A notable property of the dispatch is that the *graph can answer* both decisive and partial cases without any generative component; the LLM is invoked only in the AMBIGUOUS branch.

### 4.7.3 Generative fallback and the no-match guard

The generative component is an Ollama-served LLM (`llama3.1:8b`) contacted over the local HTTP endpoint, conditioned on the query and the top retrieved candidates. Its role is strictly circumscribed. In the skip path — the case in which the user declines to provide further detail — the answer generator deliberately *discards* the LLM's speculative response and returns a fixed, conservative directive instructing the user to provide richer fault information. This design encodes the thesis' principal anti-hallucination commitment: in the absence of sufficient graph evidence, the system prefers a truthful refusal over a fluent but ungrounded guess. The version history records exactly this transition — an earlier revision allowed the LLM to answer ambiguous queries directly, and the observed hallucination behaviour led to the no-match guard.

## 4.8 Implementation

### 4.8.1 Module organisation

The implementation is organised into a construction module set and a decision package. The construction modules implement, in order, extraction and graph build; node normalisation; embedding and index population; structural quality and schema validation; and visualisation. The decision package implements the five online stages as numbered modules (`00_score_calibrator`, `02_confidence_scorer`, `03_reasoning_path`, `04_answer_generator`), chained by a single orchestrator entry point that also guarantees a stable working directory so that relative artefact paths resolve regardless of invocation site. The numbering intentionally reserves the gaps of the original design lineage; the calibrator, for example, supersedes an earlier community-expansion stage whose function was absorbed into the retrieval layer.

### 4.8.2 Persistent artefacts

The pipeline persists the following artefacts:

- `data/processed/hierarchical_graph.json` — the normalised, community-labelled graph (node-link form);
- `data/processed/community_map.json` — per-community membership, category sets, and multi-category flags;
- `data/processed/extraction.json` — the raw node/edge extraction with provenance fields;
- `data/chroma_db` — the persistent vector index (`automotive_kg` collection, 431 embedded nodes);
- `graphify-out/` — graph visualisations, the community report, and structural analyses.

Embedding is performed once over the 431 indexable nodes, in batches of 100, with a self-test query set run at build time to validate the index before it is used.

### 4.8.3 Visualisation

Two complementary visual artefacts are produced. The first is a force-directed interactive rendering (D3-based) in which nodes are coloured by community, sized by node type, and connected by relation-coloured edges, with a legend, tooltips, and pan/zoom interaction. The second is a structural tree/call-flow view generated from the graph toolkit, suited to hierarchical navigation. Both are static HTML artefacts intended for inspection rather than for operation; the interactive interface of record is the diagnostic application described next.

### 4.8.4 Interactive interface

A Streamlit application exposes the pipeline to an interactive user. The interface renders the decision mode colour-coded and implements the full clarification loop: an AMBIGUOUS result presents the guided clarifying question with a free-text follow-up or a skip control; an INFERRED-intermediate result presents the unconfirmed symptom set as toggles, allowing the user to refine the diagnosis by confirming additional symptoms; and an EXTRACTED (or refined) result renders the diagnosis with matched symptoms, ordered diagnostic steps, and the reasoning chain, alongside a raw-answer panel for inspection. The interface is deliberately a thin presentation layer over the pipeline contract — it contains no retrieval or scoring logic of its own.

## 4.9 Evaluation and Discussion

### 4.9.1 Structural evaluation of the artefact

The constructed graph is evaluated against an eleven-point structural scorecard spanning four concerns: *coverage* (all 13 categories and 98 subcategories present; symptom and diagnosis-step coverage above 80%); *hierarchy integrity* (every category has subcategories, every subcategory has symptoms and steps); *community quality* (more than 10 communities, a majority multi-category, with plausible sizes); *cross-category signal* (more than 100 cross-category edges); and *cleanliness* (no duplicate labels). The artefact satisfies the scorecard in full. This is a property of the corpus as much as of the construction pipeline: the curated, schema-faithful source records guarantee that the extraction step is lossless, so the evaluation measures the *fidelity* of graph construction rather than the *accuracy* of an extraction model — a boundary that is stated explicitly so that the scorecard is not over-interpreted.

### 4.9.2 Behavioural evaluation of retrieval

Retrieval behaviour is characterised through canonical probe queries spanning the three modes. For a decisive query ("soft brake pedal"), the vector path returns near-identical symptom nodes whose phrase differs from the query, the structural path adds the brake hierarchy context, and the community path contributes the ABS-and-brake community; the resulting top candidates are corroborated across all three paths, receive the maximum boost, and clear the EXTRACTED threshold after calibration and path-weighting. For a plausible but under-specified query, the top candidate is corroborated by only a subset of paths, its effective confidence lands in the INFERRED band, and the reasoning path surfaces unconfirmed sibling symptoms. For an over-generic query, no candidate accumulates sufficient multi-path support and the system refuses, generating a clarifying question. The mode boundaries, in short, behave as designed: mode is a monotone function of the *quantity and independence* of corroborating evidence.

### 4.9.3 Discussion of design decisions

Three decisions merit discussion. First, the separation of *ranking* from *decision* is the pipeline's most consequential architectural choice: it allows the retrieval layer to be improved (better models, more paths) without perturbing the semantics of the mode thresholds, because the two are decoupled by the calibration and path-weighting stages. Second, the *guided clarification* design treats an ambiguous query as a dialogue event rather than a failure: the system states what it matched and asks for exactly the information that would discriminate among its candidates. Third, the *no-match guard* subordinates fluency to truthfulness in the AMBIGUOUS-skip branch. The recorded history of the project shows that this guard was introduced reactively, after an unguarded LLM fallback produced a hallucinated diagnosis; its retention thereafter is a deliberate policy, and it is identified here as the pipeline's single most important correctness safeguard.

### 4.9.4 Limitations and threats to validity

The evaluation is subject to several limitations. (1) *Corpus-bound fidelity*: because the corpus is already structured, the construction pipeline has not been stress-tested on unstructured source text, and the structural scorecard does not measure extraction accuracy in that setting. (2) *Deferred retrieval sophistication*: the implementation explicitly omits intent classification, multi-query expansion, fuzzy/typo correction, and category-aware query expansion; these were scoped out of this version and are documented as such in the retrieval-layer specification, so queries exhibiting surface noise are expected to underperform. (3) *Lexical symptom confirmation*: the matched/unconfirmed symptom partition uses word overlap, which is brittle to paraphrase; the code isolates this decision in a single function precisely so that semantic matching can be substituted without architectural change. (4) *Specification drift*: the fusion-offset discrepancy between the retrieval specification and its implementation (Section 4.5.4) demonstrates that the calibrated confidence model depends on implementation details that are easy to mis-specify, and it motivates the warning recorded against the calibration offsets. (5) *No quantitative gold-standard evaluation*: retrieval ranking and mode correctness are demonstrated behaviourally, not measured against a labelled test set; no ground-truth query corpus exists for this version, so precision/recall and mode-accuracy figures are not reported. (6) *LLM dependence on local infrastructure*: the generative fallback requires a local Ollama service, and its contribution is confined to a narrow branch; the pipeline degrades to the graph answer path without it. (7) *Threat to external validity*: the knowledge substrate covers a specific automotive taxonomy, and the numeric thresholds (0.75, 0.55; the path factors; the fusion boosts) were chosen by design inspection rather than by statistical optimisation over a development set — a calibration methodology that later pipeline versions revisit.

## 4.10 Summary and Outlook

This chapter has presented Vehicle-Fault-KG, a knowledge-graph-driven diagnostic pipeline that grounds automotive fault reasoning in an explicitly constructed and traceable artefact. Its contributions are: (i) a lossless, provenance-preserving construction of a hierarchical fault graph with explicit extracted/inferred edge semantics; (ii) a hybrid retrieval layer in which independent semantic, structural, and community evidence is fused by an agreement-rewarding scheme; (iii) a two-stage confidence model that first calibrates cross-source scores and then discounts correlated evidence by path-awareness; and (iv) a three-mode operational semantics that partitions the answer space into decisive, refinement-seeking, and refusal behaviours, with a hard anti-hallucination guard on the generative fallback.

The chapter also fixes the architectural skeleton that subsequent pipeline versions inherit: staged construction; a retrieval-to-decision interface defined by an explicit candidate contract; confidence as the sole arbiter of commitment; and the principle that the graph, not the generative model, is the source of diagnostic truth. The two most consequential developments in later versions are anticipated here. First, the *three-mode* decision space — in which INFERRED occupies an intermediate state — is revisited; the evidence that INFERRED answers are perennially treated by users as intermediate prompts rather than as diagnoses motivates a simplification to two operational modes in which every non-extracted result is handled by a single clarification-and-refinement loop. Second, the *confidence model* is re-grounded: the heuristic path factors of this version are replaced, in later versions, by a confidence composition that incorporates time-series sensor evidence as an independent evidential source, re-weighting knowledge-graph support, fault-mapping agreement, and sensor corroboration. Vehicle-Fault-KG thus stands as the definitive statement of the graph-only version of the pipeline — the baseline against which the sensor-integrated architecture is measured.
