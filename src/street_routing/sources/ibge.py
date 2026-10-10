"""IBGE data sources (Brazilian public data only), downloaded once and cached under ``data/raw/ibge``.

* Boundaries: the IBGE *malhas* API v3 (states and municipalities, quality "minima" for the web) and the
  *localidades* API v1 for names and codes.
* Neighbourhoods: *Bairros*, Censo 2022, per state (shapefile).
* Streets: *Base de Faces de Logradouros*, Censo 2022, per state (one shapefile per municipality inside).
"""

from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd

API = "https://servicodados.ibge.gov.br/api"
GEOFTP = "https://geoftp.ibge.gov.br"
FACES_URL = (f"{GEOFTP}/recortes_para_fins_estatisticos/malha_de_setores_censitarios/censo_2022/"
             "base_de_faces_de_logradouros_versao_2022_censo_demografico/shp/{uf}_faces_de_logradouros_2022_shp.zip")
BAIRROS_URL = (f"{GEOFTP}/organizacao_do_territorio/malhas_territoriais/malhas_de_setores_censitarios__divisoes_"
               "intramunicipais/censo_2022/bairros/shp/UF/{uf}_bairros_CD2022.zip")
GEOJSON = "application/vnd.geo%2Bjson"


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=300) as r:
            data = r.read()
        if data[:2] == b"\x1f\x8b":  # the IBGE API answers gzip-compressed even when not asked to
            data = gzip.decompress(data)
        tmp.write_bytes(data)
        tmp.replace(path)
    return path


def _json(url: str, path: Path) -> dict | list:
    return json.loads(_download(url, path).read_text(encoding="utf-8"))


class IbgeSource:
    def __init__(self, cache: Path) -> None:
        self.cache = cache

    def states(self) -> list[dict]:
        """[{"id": 41, "sigla": "PR", "nome": "Paraná", ...}, ...]"""
        return _json(f"{API}/v1/localidades/estados", self.cache / "api" / "estados.json")

    def municipalities(self, uf_id: int) -> list[dict]:
        return _json(f"{API}/v1/localidades/estados/{uf_id}/municipios",
                     self.cache / "api" / f"municipios_{uf_id}.json")

    def states_geojson(self) -> dict:
        url = f"{API}/v3/malhas/paises/BR?formato={GEOJSON}&qualidade=minima&intrarregiao=UF"
        return _json(url, self.cache / "api" / "malha_BR_UF.geojson")

    def municipalities_geojson(self, uf_id: int) -> dict:
        url = f"{API}/v3/malhas/estados/{uf_id}?formato={GEOJSON}&qualidade=minima&intrarregiao=municipio"
        return _json(url, self.cache / "api" / f"malha_{uf_id}_municipios.geojson")

    def faces(self, uf: str, municipality: str) -> gpd.GeoDataFrame:
        archive = _download(FACES_URL.format(uf=uf), self.cache / f"{uf}_faces_de_logradouros_2022_shp.zip")
        shp = self.cache / uf / f"{municipality}_faces_de_logradouros_2022.shp"
        if not shp.exists():
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    if name.startswith(f"{uf}/{municipality}_"):
                        z.extract(name, self.cache)
        return gpd.read_file(shp)

    def bairros(self, uf: str, municipality: str) -> gpd.GeoDataFrame:
        """Neighbourhoods of the municipality; empty where IBGE publishes none (the Federal District has
        administrative regions, not bairros, and no file)."""
        try:
            archive = _download(BAIRROS_URL.format(uf=uf), self.cache / f"{uf}_bairros_CD2022.zip")
        except urllib.error.HTTPError as err:
            if err.code != 404:
                raise
            return gpd.GeoDataFrame({"CD_BAIRRO": [], "NM_BAIRRO": []}, geometry=[], crs="EPSG:4674")
        shp = self.cache / f"{uf}_bairros_CD2022.shp"
        if not shp.exists():
            zipfile.ZipFile(archive).extractall(self.cache)
        df = gpd.read_file(shp)
        return df[df["CD_MUN"].astype(str) == str(municipality)]
