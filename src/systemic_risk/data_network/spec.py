"""The canonical, layered system specification.

``NetworkSpec`` is the frozen source-of-truth object owned by part A. It separates what is
*empirically real* (marginals, equity-return correlation, balance-sheet totals) from what
is *reconstructed* under a named method (the bilateral exposure matrix — real bilateral
data is confidential), records a documented ``FeatureSchema`` (what each field means and
which consumer may see it) and ``Provenance`` (source + fit params + content hash), and:

- **round-trips losslessly** via :meth:`to_json` / :meth:`from_json`;
- **assembles down** into the flat :class:`systemic_risk.spec.SystemSpec` that parts
  B/C/D already consume, via :meth:`to_system_spec`;
- exposes **consumer-scoped views** via :meth:`view_for` (the generators see marginals +
  correlation; the simulator sees exposures + buffers; visualization sees everything).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from systemic_risk.spec import SystemSpec

# Consumer tags used by the feature-visibility contract.
GENERATOR = "generator"          # B (copula) and C (entangled) baselines
SIMULATOR = "simulator"          # D, the cascade engine
VISUALIZATION = "visualization"  # the network/cluster plot
ALL_CONSUMERS = (GENERATOR, SIMULATOR, VISUALIZATION)


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureField:
    """One field in the canonical spec: its meaning, level, and who may read it."""

    name: str
    level: str            # "node" | "edge" | "system"
    dtype: str            # "float" | "categorical" | "matrix"
    description: str
    visible_to: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "dtype": self.dtype,
            "description": self.description,
            "visible_to": list(self.visible_to),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeatureField":
        return cls(
            name=d["name"],
            level=d["level"],
            dtype=d["dtype"],
            description=d["description"],
            visible_to=tuple(d["visible_to"]),
        )


@dataclass(frozen=True)
class FeatureSchema:
    """The documented field schema — feature definitions + per-consumer visibility."""

    fields: tuple[FeatureField, ...]

    def names_visible_to(self, consumer: str) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if consumer in f.visible_to)

    def to_dict(self) -> dict[str, Any]:
        return {"fields": [f.to_dict() for f in self.fields]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeatureSchema":
        return cls(fields=tuple(FeatureField.from_dict(f) for f in d["fields"]))

    @classmethod
    def default(cls) -> "FeatureSchema":
        """The frozen feature contract for the bank-network spec."""
        return cls(
            fields=(
                FeatureField("node_id", "node", "categorical",
                             "Stable institution identifier.", ALL_CONSUMERS),
                FeatureField("name", "node", "categorical",
                             "Human-readable institution name.", (VISUALIZATION,)),
                FeatureField("node_type", "node", "categorical",
                             "Institution class (bank/insurer/...).", ALL_CONSUMERS),
                FeatureField("business_type", "node", "categorical",
                             "Business model (universal/investment/regional/custodian).",
                             (VISUALIZATION,)),
                FeatureField("region", "node", "categorical",
                             "Geographic region.", (VISUALIZATION,)),
                FeatureField("rating", "node", "categorical",
                             "Public S&P long-term issuer rating.", (VISUALIZATION,)),
                FeatureField("marginal_default_prob", "node", "float",
                             "p_i: 1-year marginal default probability from rating.",
                             (GENERATOR, VISUALIZATION)),
                FeatureField("capital_buffer", "node", "float",
                             "Tier-1-style loss-absorbing buffer used by the cascade.",
                             (SIMULATOR, VISUALIZATION)),
                FeatureField("interbank_assets", "node", "float",
                             "Total interbank claims (row-sum constraint for edges).",
                             ()),
                FeatureField("interbank_liabilities", "node", "float",
                             "Total interbank liabilities (col-sum constraint for edges).",
                             ()),
                FeatureField("correlation_matrix", "system", "matrix",
                             "Equity-return latent correlation (copula dependency target).",
                             (GENERATOR, VISUALIZATION)),
                FeatureField("exposure_matrix", "edge", "matrix",
                             "W[i,j] = loss to i if j defaults (reconstructed bilateral).",
                             (SIMULATOR, VISUALIZATION)),
                FeatureField("cluster", "node", "categorical",
                             "Community label from detection on the network.",
                             (VISUALIZATION,)),
            )
        )


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Provenance:
    """Source, fit parameters, and a content hash for auditability."""

    source: str
    fit_params: Mapping[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "fit_params": dict(self.fit_params),
            "content_hash": self.content_hash,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Provenance":
        return cls(
            source=d["source"],
            fit_params=dict(d.get("fit_params", {})),
            content_hash=d.get("content_hash", ""),
            notes=d.get("notes", ""),
        )


# --------------------------------------------------------------------------- #
# Empirical (frozen ground truth)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmpiricalLayer:
    """Frozen empirical ground truth — round-trips exactly.

    ``node_attributes`` carries the real categorical features (type/rating/region/...);
    ``node_totals`` holds total-asset scale per node; ``interbank_assets`` /
    ``interbank_liabilities`` are the marginal constraints fed to the reconstruction.
    """

    node_ids: tuple[str, ...]
    marginals: tuple[float, ...]
    correlation_matrix: np.ndarray
    capital_buffers: tuple[float, ...]
    interbank_assets: tuple[float, ...]
    interbank_liabilities: tuple[float, ...]
    node_totals: Mapping[str, float]
    node_attributes: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        n = len(self.node_ids)
        corr = _finite_array(self.correlation_matrix, "correlation_matrix")
        if corr.shape != (n, n):
            raise ValueError("correlation_matrix must be (n, n)")
        if not np.allclose(corr, corr.T, atol=1e-8):
            raise ValueError("correlation_matrix must be symmetric")
        for name, seq in (
            ("marginals", self.marginals),
            ("capital_buffers", self.capital_buffers),
            ("interbank_assets", self.interbank_assets),
            ("interbank_liabilities", self.interbank_liabilities),
        ):
            if len(seq) != n:
                raise ValueError(f"{name} must have length {n}")
        if any(not 0.0 <= p <= 1.0 for p in self.marginals):
            raise ValueError("marginals must lie in [0, 1]")
        object.__setattr__(self, "correlation_matrix", corr)

    @property
    def n(self) -> int:
        return len(self.node_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_ids": list(self.node_ids),
            "marginals": list(self.marginals),
            "correlation_matrix": self.correlation_matrix.tolist(),
            "capital_buffers": list(self.capital_buffers),
            "interbank_assets": list(self.interbank_assets),
            "interbank_liabilities": list(self.interbank_liabilities),
            "node_totals": dict(self.node_totals),
            "node_attributes": {k: dict(v) for k, v in self.node_attributes.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EmpiricalLayer":
        return cls(
            node_ids=tuple(d["node_ids"]),
            marginals=tuple(float(x) for x in d["marginals"]),
            correlation_matrix=np.asarray(d["correlation_matrix"], dtype=float),
            capital_buffers=tuple(float(x) for x in d["capital_buffers"]),
            interbank_assets=tuple(float(x) for x in d["interbank_assets"]),
            interbank_liabilities=tuple(float(x) for x in d["interbank_liabilities"]),
            node_totals={k: float(v) for k, v in d["node_totals"].items()},
            node_attributes={k: dict(v) for k, v in d["node_attributes"].items()},
        )


# --------------------------------------------------------------------------- #
# Reconstructed (swappable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReconstructedLayer:
    """Swappable bilateral exposures — round-trips up to its method tag."""

    edge_matrix: np.ndarray
    method: str
    method_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        edges = _finite_array(self.edge_matrix, "edge_matrix")
        if edges.ndim != 2 or edges.shape[0] != edges.shape[1]:
            raise ValueError("edge_matrix must be square")
        if np.any(edges < 0):
            raise ValueError("edge_matrix must be nonnegative")
        if np.any(np.diag(edges) != 0):
            raise ValueError("edge_matrix diagonal must be zero")
        object.__setattr__(self, "edge_matrix", edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_matrix": self.edge_matrix.tolist(),
            "method": self.method,
            "method_params": dict(self.method_params),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReconstructedLayer":
        return cls(
            edge_matrix=np.asarray(d["edge_matrix"], dtype=float),
            method=d["method"],
            method_params=dict(d.get("method_params", {})),
        )


# --------------------------------------------------------------------------- #
# Consumer-scoped view
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpecView:
    """A read-only projection of the spec containing only a consumer's visible fields."""

    consumer: str
    fields: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def keys(self) -> Any:
        return self.fields.keys()


# --------------------------------------------------------------------------- #
# The canonical spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NetworkSpec:
    """Layered, frozen canonical spec. Assembles down into the flat ``SystemSpec``."""

    empirical: EmpiricalLayer
    reconstructed: ReconstructedLayer
    clusters: tuple[int, ...]
    feature_schema: FeatureSchema
    provenance: Provenance

    def __post_init__(self) -> None:
        n = self.empirical.n
        if self.reconstructed.edge_matrix.shape != (n, n):
            raise ValueError("reconstructed edge_matrix must match the empirical node count")
        if len(self.clusters) != n:
            raise ValueError("clusters must have one label per node")

    @property
    def n(self) -> int:
        return self.empirical.n

    # ---- content hash ---------------------------------------------------- #
    def compute_content_hash(self) -> str:
        """Deterministic SHA-256 over the substantive payload (excludes the hash itself)."""
        payload = json.dumps(
            {
                "empirical": self.empirical.to_dict(),
                "reconstructed": self.reconstructed.to_dict(),
                "clusters": list(self.clusters),
                "feature_schema": self.feature_schema.to_dict(),
                "source": self.provenance.source,
                "fit_params": dict(self.provenance.fit_params),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_content_hash(self) -> "NetworkSpec":
        """Return a copy whose provenance carries the freshly computed content hash."""
        prov = Provenance(
            source=self.provenance.source,
            fit_params=self.provenance.fit_params,
            content_hash=self.compute_content_hash(),
            notes=self.provenance.notes,
        )
        return NetworkSpec(self.empirical, self.reconstructed, self.clusters,
                           self.feature_schema, prov)

    # ---- consumer views -------------------------------------------------- #
    def view_for(self, consumer: str) -> SpecView:
        """Project the spec onto the fields ``consumer`` is allowed to see."""
        emp = self.empirical
        available: dict[str, Any] = {
            "node_id": list(emp.node_ids),
            "name": [emp.node_attributes[i].get("name", i) for i in emp.node_ids],
            "node_type": [emp.node_attributes[i].get("node_type", "bank") for i in emp.node_ids],
            "business_type": [emp.node_attributes[i].get("business_type", "") for i in emp.node_ids],
            "region": [emp.node_attributes[i].get("region", "") for i in emp.node_ids],
            "rating": [emp.node_attributes[i].get("rating", "") for i in emp.node_ids],
            "marginal_default_prob": list(emp.marginals),
            "capital_buffer": list(emp.capital_buffers),
            "interbank_assets": list(emp.interbank_assets),
            "interbank_liabilities": list(emp.interbank_liabilities),
            "correlation_matrix": emp.correlation_matrix,
            "exposure_matrix": self.reconstructed.edge_matrix,
            "cluster": list(self.clusters),
        }
        visible = self.feature_schema.names_visible_to(consumer)
        return SpecView(consumer=consumer, fields={k: available[k] for k in visible if k in available})

    # ---- assemble down to the B/C/D contract ----------------------------- #
    def to_system_spec(self) -> SystemSpec:
        """Emit the flat ``SystemSpec`` consumed by generators / simulator / evaluation."""
        emp = self.empirical
        return SystemSpec(
            node_names=[emp.node_attributes[i].get("name", i) for i in emp.node_ids],
            node_types=[emp.node_attributes[i].get("node_type", "bank") for i in emp.node_ids],
            exposure_matrix=self.reconstructed.edge_matrix,
            capital_buffers=np.asarray(emp.capital_buffers, dtype=float),
            marginal_default_probs=np.asarray(emp.marginals, dtype=float),
            target_pairwise_corr=emp.correlation_matrix,
            clusters=[f"community_{c}" for c in self.clusters],
            metadata={
                "name": "Bank exposure network (real anchor)",
                "correlation_space": str(
                    self.provenance.fit_params.get(
                        "correlation_space",
                        "latent_gaussian",
                    )
                ),
                "source": self.provenance.source,
                "content_hash": self.provenance.content_hash or self.compute_content_hash(),
                "reconstruction_method": self.reconstructed.method,
                "reconstruction_params": dict(self.reconstructed.method_params),
                "fit_params": dict(self.provenance.fit_params),
                "node_ids": list(emp.node_ids),
                "ratings": [emp.node_attributes[i].get("rating", "") for i in emp.node_ids],
                "regions": [emp.node_attributes[i].get("region", "") for i in emp.node_ids],
                "business_types": [emp.node_attributes[i].get("business_type", "") for i in emp.node_ids],
                "n_communities": len(set(self.clusters)),
                "feature_schema": self.feature_schema.to_dict(),
                "provenance_notes": self.provenance.notes,
            },
        )

    # ---- (de)serialization ----------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "empirical": self.empirical.to_dict(),
            "reconstructed": self.reconstructed.to_dict(),
            "clusters": list(self.clusters),
            "feature_schema": self.feature_schema.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NetworkSpec":
        return cls(
            empirical=EmpiricalLayer.from_dict(d["empirical"]),
            reconstructed=ReconstructedLayer.from_dict(d["reconstructed"]),
            clusters=tuple(int(c) for c in d["clusters"]),
            feature_schema=FeatureSchema.from_dict(d["feature_schema"]),
            provenance=Provenance.from_dict(d["provenance"]),
        )

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "NetworkSpec":
        return cls.from_dict(json.loads(s))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "NetworkSpec":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
