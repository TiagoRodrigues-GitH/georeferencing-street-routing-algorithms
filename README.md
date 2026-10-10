# Georeferencing Street Routing Algorithms

**From satellite images to routes:** detect streets and buildings in Londrina (PR, Brazil) in CBERS-4A satellite
images, turn the streets into a road graph, and compare classic, heuristic and evolutionary path-finding
algorithms on it, with an interactive map that animates each algorithm between an origin and a destination.

| | |
|---|---|
| **Author** | Tiago Rodrigues · Universidade Tecnológica Federal do Paraná (UTFPR) |
| **Started** | 2026-10-04 |
| **Status** | Milestone 1 in progress: street graphs built from IBGE block faces for six areas (Londrina, Curitiba, Florianópolis, Brasília, São Paulo, ABC Paulista; 39,000 km of streets); routing demo in the portfolio |
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
python -m street_routing.export.web --out outputs/web_assets                 # all six areas (~10 min)
python -m street_routing.export.web --area 3550308 --out outputs/web_assets  # one area; the others are kept
```

The export downloads the IBGE files once (`data/raw/ibge`), builds the street graph of each routed area
([export/areas.py](src/street_routing/export/areas.py)) and writes the files of the web map: `places.json` (search
index of every state, municipality and neighbourhood of the routed areas, and the list of routed areas),
`states.geojson`, `municipalities/<state>.geojson`, `bairros/<area>.geojson` (where IBGE publishes neighbourhoods)
and `graph/<area>.json`.

| Area | Intersections | Streets | Parts | File |
|---|---|---|---|---|
| Londrina (PR) | 10,640 | 2,003 km | 1 | 1.8 MB |
| Curitiba (PR) | 27,242 | 4,655 km | 1 | 5.0 MB |
| Florianópolis (SC) | 9,405 | 1,535 km | 5 | 1.5 MB |
| Brasília (DF) | 57,674 | 9,265 km | 75 | 10.5 MB |
| São Paulo (SP) | 112,928 | 16,201 km | 1 | 20.1 MB |
| ABC Paulista (7 municipalities, SP) | 30,828 | 5,012 km | 1 | 5.5 MB |

Large cities are rasterised in 8 km tiles with a 400 m overlap (the skeleton of each core is exact), each area is
projected in its own UTM zone, and every connected part with at least 20 km of streets is kept: in Florianópolis and
Brasília, bridges and highways have no block faces, so the network stays in parts and no link is invented.

**How the graph is built.** IBGE publishes block faces, not street centre lines, and the faces stop at the block
corners. Each face is buffered by 8 m and rasterised at 2 m; the holes left at crossings are filled; the corridor is
thinned to a skeleton and the skeleton becomes a graph (spurs pruned, junctions split by the thinning merged, edges
named after the nearest face). Roads with no facing blocks (avenues along lakes and parks, bridges) leave districts
cut off: parts closer than 100 m are joined by straight estimated links, marked as such. The inner sides of an
avenue's carriageways face no block either, so the openings of wide medians are missing: junctions under 100 m apart
whose road distance is over 10x their gap are joined too, when the link crosses no block face and no street (pairs
150-600 m apart needing a detour over 5x: 13.9% -> 3.8%). Londrina: 10,640 junctions, 2,003 km in the routable
network, 143 estimated links totalling 7.75 km (0.4%).

## Data

| Data | Publisher | Use | Licence |
|---|---|---|---|
| CBERS-4A WPM: 2 m fused RGB, 8 m multispectral (incl. near infrared) | INPE, STAC catalogue at data.inpe.br | images | CC BY 4.0 |
| Base de Faces de Logradouros, Censo 2022 (street lines per municipality) | IBGE | road labels, routing graph, reference for APLS | public (cite IBGE) |
| Municipal boundary (Malha Municipal) | IBGE | area of interest | public (cite IBGE) |
| Building labels | hand-annotated sample on CBERS-4A tiles | building class | this project |
| `oneway` tags of OpenStreetMap ways (Overpass API) | OpenStreetMap contributors | direction of one-way streets only, matched to the IBGE edges | ODbL 1.0 (attribution: "© OpenStreetMap contributors") |

One dataset family (INPE images + IBGE vectors) keeps the article's data reproducible and citable; OpenStreetMap
supplies only the direction of one-way streets, which no Brazilian public source publishes (ADR 3, 13, 14).
The graphs with directions are a derived database under the ODbL: they credit OpenStreetMap and are shared
under the same licence.

```
python -m street_routing.sources.osm          # fetch and cache the one-way ways of every area (Overpass, polite pauses)
python -m street_routing.export.oneway_web    # match them to the web graphs and write payload["oneway"]
```

| Area | OSM one-way ways | Edges matched | Reverted by the repair | One-way edges |
|---|---:|---:|---:|---:|
| Londrina | 4,414 | 2,565 of 16,526 | 119 | 2,446 |
| Curitiba | 13,766 | 5,609 of 41,289 | 56 | 5,553 |
| Florianópolis | 3,974 | 1,121 of 11,829 | 85 | 1,036 |
| Brasília | 26,499 | 2,324 of 84,650 | 498 | 1,826 |
| São Paulo | 76,103 | 21,565 of 165,108 | 170 | 21,395 |
| ABC Paulista | 19,903 | 6,371 of 44,464 | 166 | 6,205 |

Brasília matches least: its wide avenues and interchanges lie farther than the 12 m tolerance from the skeleton
of the IBGE corridors, so many of them stay two-way.
