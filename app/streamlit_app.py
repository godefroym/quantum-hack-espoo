from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / "outputs" / ".matplotlib"
XDG_CACHE = ROOT / "outputs" / ".cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from systemic_risk.data import make_synthetic_system
from systemic_risk.evaluation import EvaluationHarness
from systemic_risk.generators import (
    BernoulliGenerator,
    EntangledPQCGenerator,
    GaussianCopulaGenerator,
    StudentTCopulaGenerator,
)
from systemic_risk.visualization import plot_financial_network


st.set_page_config(page_title="Quantum Systemic Stress Scenario Discovery", layout="wide")

st.title("Quantum Systemic Stress Scenario Discovery")

n = st.sidebar.slider("Institutions", min_value=12, max_value=20, value=16)
n_samples = st.sidebar.slider("Samples", min_value=200, max_value=3000, value=1000, step=200)
seed = st.sidebar.number_input("Seed", min_value=0, value=7, step=1)

spec = make_synthetic_system(n=n, seed=int(seed))
generators = [
    BernoulliGenerator(),
    GaussianCopulaGenerator(),
    StudentTCopulaGenerator(df=4.0),
    EntangledPQCGenerator(layers=2),
]

left, right = st.columns([1, 1])
with left:
    st.subheader("Financial network")
    fig = plot_financial_network(spec)
    st.pyplot(fig)

with right:
    st.subheader("Cascade comparison")
    if st.button("Run comparison"):
        harness = EvaluationHarness(spec, n_samples=n_samples, seed=int(seed) + 1000)
        frame = harness.to_frame(harness.run(generators))
        st.dataframe(frame, use_container_width=True)
