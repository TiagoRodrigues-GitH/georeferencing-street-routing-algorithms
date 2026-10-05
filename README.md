# Georeferencing Street Routing Algorithms

**From satellite images to routes:** detect streets and buildings in Londrina (PR, Brazil) in CBERS-4A satellite
images, turn the streets into a road graph, and compare classic, heuristic and evolutionary path-finding
algorithms on it, with an interactive map that animates each algorithm between an origin and a destination.

| | |
|---|---|
| **Author** | Tiago Rodrigues · Universidade Tecnológica Federal do Paraná (UTFPR) |
| **Started** | 2026-10-04 |
| **Status** | Milestone 1 in progress: Londrina street graph built from IBGE block faces (2,000 km, 10,640 intersections); routing demo in the portfolio |
| **Context** | AI Residency project (Georeferencing) |
| **Stack (planned)** | Python 3.12 · rasterio · GeoPandas · NetworkX · PyTorch · DEAP/pymoo · Optuna · TypeScript + MapLibre |

> **Resumo (PT).** Detectar ruas e edificações em imagens do satélite CBERS-4A (INPE, 2 m) de Londrina, separar
> ruas de casas, transformar as ruas em um grafo e comparar algoritmos de caminho (Dijkstra, A*, colônia de
> formigas, algoritmo genético, NSGA-II...) quanto à qualidade da rota e ao tempo, com uma página web que mostra o
> mapa, recebe origem e destino e anima cada algoritmo, com hiperparâmetros ajustáveis. Só dados públicos
> brasileiros: imagens do INPE e ruas do IBGE (Censo 2022). Pipeline em três etapas: pré-processamento,
> processamento e pós-processamento.

## Questions

1. How well can streets and buildings be separated in 2 m CBERS-4A images, using the IBGE street lines as labels?
2. How close is the road graph extracted from the images to the IBGE street network (APLS metric), and how much do
   the extraction errors change the routes?
3. Which path-finding algorithms give the best trade-off between route quality and computing time on a real city
   network: exact (Dijkstra, A*, bidirectional, contraction hierarchies) or metaheuristic (ant colony, genetic
   algorithm, simulated annealing; NSGA-II for several objectives)?

## Pipeline

| Stage | What it does |
|---|---|
| **Pre-processing** | Londrina boundary and street lines (IBGE), CBERS-4A image tiles (INPE); reprojection to SIRGAS 2000 / UTM 22S; tiling, normalisation, cloud mask; labels rasterised from the IBGE streets and a hand-annotated building sample |
| **Processing** | Semantic segmentation of road / building / background ([neural network](docs/NEURAL_NETWORK.md)); road mask to graph (skeleton, junctions, simplification); building polygons |
| **Post-processing** | Graph cleaning and comparison with the IBGE network (APLS); path-finding algorithms on both graphs; metrics and statistics; the web map with animations |

Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The neural network and every hyperparameter you can change,
with its expected effect: [docs/NEURAL_NETWORK.md](docs/NEURAL_NETWORK.md), [configs/segmentation.yaml](configs/segmentation.yaml),
[configs/routing.yaml](configs/routing.yaml).

## Milestones

1. **Routing on the IBGE street network** (no images needed): algorithm library, benchmark on many
   origin-destination pairs, web map with animations and hyperparameter sliders. First public demo.
2. **Imagery and segmentation**: CBERS-4A tiles of Londrina, labels, segmentation models and hyperparameter search (GPU).
3. **Extracted graph**: graph from the road mask, APLS against the IBGE network, routing on the extracted graph.
4. **Write-up**: results, paper draft, dataset release (CBERS-4A tiles + IBGE labels for Londrina).

## How to run (milestone 1)

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e .[dev]     # Windows; .venv/bin on Linux
python -m pytest                                                            # synthetic skeletons and crossroads
python -m street_routing.export.web --uf PR --municipality 4113700 --out outputs/web_assets
```

The export downloads the IBGE files once (`data/raw/ibge`), builds the street graph and writes the files of the
web map: `places.json` (search index of every state, municipality and Londrina neighbourhood), `states.geojson`,
`municipalities/<state>.geojson`, `bairros/4113700.geojson` and `graph/4113700.json`.

**How the graph is built.** IBGE publishes block faces, not street centre lines, and the faces stop at the block
corners. Each face is buffered by 8 m and rasterised at 2 m; the holes left at crossings are filled; the corridor is
thinned to a skeleton and the skeleton becomes a graph (spurs pruned, junctions split by the thinning merged, edges
named after the nearest face). Roads with no facing blocks (avenues along lakes and parks, bridges) leave districts
cut off: parts closer than 100 m are joined by straight estimated links, marked as such. Londrina: 2,047 km of
corridor centre line, 1,996 km in the routable network (97.5%), 17 estimated links totalling 0.56 km.

## Data (Brazilian public sources only)

| Data | Publisher | Use | Licence |
|---|---|---|---|
| CBERS-4A WPM: 2 m fused RGB, 8 m multispectral (incl. near infrared) | INPE, STAC catalogue at data.inpe.br | images | CC BY 4.0 |
| Base de Faces de Logradouros, Censo 2022 (street lines per municipality) | IBGE | road labels, routing graph, reference for APLS | public (cite IBGE) |
| Municipal boundary (Malha Municipal) | IBGE | area of interest | public (cite IBGE) |
| Building labels | hand-annotated sample on CBERS-4A tiles | building class | this project |

One dataset family (INPE images + IBGE vectors) keeps the article's data reproducible and citable.
