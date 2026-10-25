# NEURON Computational Modelling Workshop

Hands-on introduction to NEURON in Python, using the retinal ganglion cell model of Sheasby & Fohlmeister (1999).

## Open the workshop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jseo977/UniMelb-PhD-Neuroscience-Computational-Modelling-Workshop/blob/main/run_sheasbyfohlmeister1999_workshop.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jseo977/UniMelb-PhD-Neuroscience-Computational-Modelling-Workshop/HEAD?urlpath=lab/tree/run_sheasbyfohlmeister1999_workshop.ipynb)

**Colab (recommended):** click the badge, sign in with a Google account, then run the **Setup** cells at the top
(click **Run anyway** if Colab warns the notebook was not authored by Google).
Use **File → Save a copy in Drive** if you want to keep your changes.

**Binder (backup):** click the badge and wait for the environment to build. Nothing is saved; download your notebook before closing.

## Running locally

```bash
pip install -r requirements.txt
nrnivmodl          # compile the .mod files (run in this folder)
jupyter lab run_sheasbyfohlmeister1999_workshop.ipynb
```

## Contents

| File | Purpose |
|---|---|
| `run_sheasbyfohlmeister1999_workshop.ipynb` | Workshop notebook |
| `sheasbyfohlmeister1999_utils.py` | Helper functions (electrode placement, plotting) |
| `spike.mod`, `capump.mod`, `xtra.mod` | Ion channels, Ca pump, extracellular stimulation |
| `sheasbyfohlmeister1999/MediumComplex/CTT3219B/` | Cell morphology |
| `requirements.txt`, `postBuild` | Environment for Binder / local install |
