# GovSimElect: Governance of the Commons Simulation with Elections

Please see the [original GovSim project](https://github.com/giorgiopiatti/GovSim) for details of the baseline setup for GovSim. Stay tuned for further details regarding GovSimElect or contact me directly @ ryan.art.faulkner@gmail.com.

![GovSim overview](imgs/govsim_pull_figure.png)

TODO: Update workflow above with election phases.

## Code Setup

To use the codes in this repo, first clone this repo:
    
    git clone https://github.com/rfaulkner/GovSimElect.git
    cd GovSimElect

Then, to install the full local-model stack, run:

```setup
bash ./setup.sh
```

For a lighter cloud-only setup that targets OpenAI or OpenRouter and skips the local LLM stack, use:

```setup
bash ./setup_cloud.sh
```

The cloud setup also avoids requiring `sentence_transformers`; if that package is absent, the simulation falls back to a lightweight hash-based embedding for memory retrieval.

## Frontend

The repo also includes a React frontend for browsing simulation results under [`frontend/`](/home/angeloh/Desktop/work/GovSimElect/frontend).

Install the frontend dependencies with:

```bash
cd frontend
pnpm install
```

For local development, run:

```bash
cd frontend
pnpm dev
```

This starts both:
- the Vite frontend on `http://localhost:5173`
- the analysis backend on `http://localhost:8050`

To build the frontend bundle:

```bash
cd frontend
pnpm build
```

If you want to run the analysis backend separately without the Vite dev server:

```bash
uv run python3 -m simulation.analysis.app
```
