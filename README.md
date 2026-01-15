# TOPol: Topic-Orientation Polarity

Semantic polarity in computational linguistics has traditionally been framed as sentiment along a unidimensional scale. We here challenge and advance this framing, as it oversimplifies the inherently multidimensional nature of language. We introduce TOPol (Topic-Orientation Polarity), a semi- unsupervised framework for reconstructing and interpreting multidimensional narrative polarity fields given human-on- the-loop (HoTL) defined contextual boundaries (CBs).

---

## ⚙️ Overall concepts

TOPol begins by embedding documents using a general-purpose transformer-based large language model (tLLM), followed by a neighbor-tuned UMAP projection and topic-based segmentation via Leiden partitioning. Given a CB between regimes A and B, the framework computes directional vectors between corresponding topic-boundary centroids, producing a polarity field that captures fine-grained semantic displacement for each discourse regime change. TOPol polarity field reveals CB quality as this vectorial representation enables quantification of the magnitude, direction, and semantic meaning of polarity shifts, acting as a polarity change detection tool directing HoTL CB tuning. To interpret TOPol identified polarity shifts, we use the tLLM  to compare the extreme points of each polarity vector and generate contrastive labels with estimated coverage. Robustness tests confirm that only the definition of CBs, the primary HoTL-tunable parameter, significantly modulates TOPol outputs, indicating methodological stability.

---

## 🚀 Quick Start

### 1. Create the environment and install dependencies

Launch the setup script:

```bash
source .devenv/setup.bashrc
```

This will:
- Create a new Conda environment named `topol`
- Install all required Python dependencies

---

### 2. Set up API keys

- Copy the example environment file:

```bash
cp .env.example .env
```

- Open `.env` and insert your API keys (e.g., OpenAI key for embedding/sentiment scoring)

---

### 3. Download Preprocessed Data (Recommended)

To avoid recomputing text cleaning, transformer embeddings, and sentiment labels, download the preprocessed data folder:

Data available at:  
https://osf.io/nr94j/?view_only=de5b6b40ada34c6ab5cccfaf22dd5d78

Unzip and place the `data/` folder at the project root.

---

## 📁 Project Structure

- `.devenv/` — Environment setup scripts
- `src/` — Core implementation (embedding, reduction, clustering, drift, interpretation)
- `notebooks/` — Experimental notebooks
- `data/` — Preprocessed data (optional download)
- `outputs/` — Analysis outputs.
- `.env.example` — Template for API key configuration

---
