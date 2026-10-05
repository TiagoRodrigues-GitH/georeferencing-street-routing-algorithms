# Londrina Street Routing

**From satellite images to routes:** detect streets and buildings in Londrina (PR, Brazil), turn the streets
into a road graph, and compare classic, heuristic and evolutionary path-finding algorithms on it, with an
interactive map that animates each algorithm between an origin and a destination.

| | |
|---|---|
| **Author** | Tiago Rodrigues · Universidade Tecnológica Federal do Paraná (UTFPR) |
| **Started** | 2026-10-04 |
| **Status** | Design: architecture draft, no code yet |
| **Context** | AI Residency project (Georeferencing) |
| **Stack (planned)** | Python 3.12 · rasterio · GeoPandas · OSMnx · NetworkX · PyTorch · DEAP/pymoo · TypeScript + MapLibre |

> **Resumo (PT).** Detectar ruas e edificações em imagens de satélite de Londrina (CBERS-4A, 2 m), separar
> ruas de casas, transformar as ruas em um grafo e comparar algoritmos de caminho (Dijkstra, A*, colônia de
> formigas, algoritmo genético, NSGA-II...) quanto à qualidade da rota e ao tempo, com uma página web que mostra
> o mapa, recebe origem e destino e anima cada algoritmo. Pipeline em três etapas: pré-processamento,
> processamento e pós-processamento.

## Questions

1. How well can streets and buildings be separated in 2 m Brazilian satellite imagery (CBERS-4A) with weak labels
   from OpenStreetMap and Google Open Buildings?
2. How close is the road graph extracted from the images to the OpenStreetMap graph (APLS metric), and how much do
   the extraction errors change the routes?
3. Which path-finding algorithms give the best trade-off between route quality and computing time on a real city
   network: exact (Dijkstra, A*, bidirectional, contraction hierarchies) or metaheuristic (ant colony, genetic
   algorithm, simulated annealing; NSGA-II for several objectives)?

## Pipeline

| Stage | What it does |
|---|---|
| **Pre-processing** | Londrina boundary (IBGE), CBERS-4A image tiles, OpenStreetMap roads and buildings, Google Open Buildings; reprojection to SIRGAS 2000 / UTM 22S; tiling, normalisation, cloud mask; weak labels rasterised from the vector data |
| **Processing** | Semantic segmentation of road / building / background (U-Net, DeepLabv3+, SegFormer); road mask to graph (skeleton, junctions, simplification); building polygons |
| **Post-processing** | Graph cleaning and comparison with OpenStreetMap (APLS); path-finding algorithms on both graphs; metrics and statistics; the web map with animations |

Design, modules and references: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Milestones

1. **Routing on the OpenStreetMap graph** (no images needed): algorithm library, benchmark on many origin-destination
   pairs, web map with animations. First public demo.
2. **Imagery and segmentation**: CBERS-4A tiles of Londrina, weak labels, segmentation models (GPU).
3. **Extracted graph**: graph from the road mask, APLS against OpenStreetMap, routing on the extracted graph.
4. **Write-up and GitHub release**: results, paper draft, public repository.

## Data and licences

| Data | Source | Licence |
|---|---|---|
| CBERS-4A WPM, 2 m fused RGB | INPE (Brazil), STAC catalogue at data.inpe.br | CC BY 4.0 |
| Municipal boundary | IBGE | public |
| Roads and buildings | OpenStreetMap | ODbL (attribution required) |
| Building footprints | Google Open Buildings v3 | CC BY 4.0 or ODbL |
