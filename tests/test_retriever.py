"""Retriever with a fake embedder, and the policy's retriever shortlist strategy."""
import numpy as np

from laya_agent.brain.laya_policy import LayaPolicy
from laya_agent.brain.retriever import EmbeddingRetriever
from laya_agent.config import Config
from laya_agent.models import Rect, Snapshot, UIElement
from tests.test_policy import _click_keys, _fake_predict

SYN = {"louder": "volume", "delete": "remove", "trash": "remove", "rid": "remove"}


def fake_embed(texts):
    """Bag-of-words with a tiny synonym table, unit-normalised. Enough to test plumbing."""
    vocab = sorted({SYN.get(w, w) for t in texts for w in t.lower().replace("'", " ").split()})
    rows = []
    for t in texts:
        v = np.zeros(len(vocab))
        for w in t.lower().replace("'", " ").split():
            v[vocab.index(SYN.get(w, w))] += 1
        rows.append(v / (np.linalg.norm(v) or 1))
    return np.array(rows)


def _els(names, kind="Button"):
    return [UIElement(id=i + 1, kind=kind, name=n, rect=Rect(0, i * 10, 50, i * 10 + 9)) for i, n in enumerate(names)]


def _policy(fake=None):
    fake = fake or _fake_predict("click_1")
    return LayaPolicy(Config(alias_path=None), predict=fake, retriever=EmbeddingRetriever(Config(), embed=fake_embed))


def _fine_names(policy, d):
    return {d.raw["_elements"][f"click_{i}"].name for i in d.raw["_fine"]}


def test_rank_orders_by_similarity():
    r = EmbeddingRetriever(Config(), embed=fake_embed)
    ranked = r.rank("make it louder", _els(["Play", "Volume", "Next"]))
    assert ranked[0][0].name == "Volume" and ranked[0][1] > ranked[1][1]


def test_retriever_shortlist_keeps_synonym_regardless_of_order():
    names = [f"Thing {i}" for i in range(40)] + ["Delete"]
    for order in (names, list(reversed(names))):
        policy = _policy()
        snap = Snapshot(png=b"", width=1, height=1, window_title="w", elements=_els(order))
        d = policy.decide("get rid of this file", snap, history=[])
        assert "Delete" in _fine_names(policy, d)
        assert len(d.raw["_fine"]) == policy.FINE_K


def test_retriever_needs_exactly_one_laya_pass():
    fake = _fake_predict("click_1")
    policy = _policy(fake)
    snap = Snapshot(png=b"", width=1, height=1, window_title="w", elements=_els([f"Item {i}" for i in range(60)]))
    policy.decide("pick something", snap, history=[])
    sizes = [len(_click_keys(c)) for c in fake.calls]
    assert len(fake.calls) == 1 and sizes[0] == policy.FINE_K  # retrieval fills the list, Laya only decides


def test_chunks_strategy_still_available():
    fake = _fake_predict("click_1")
    policy = LayaPolicy(Config(alias_path=None, shortlist="chunks"), predict=fake, retriever=None)
    snap = Snapshot(png=b"", width=1, height=1, window_title="w", elements=_els([f"Item {i}" for i in range(60)]))
    policy.decide("pick something", snap, history=[])
    assert len(fake.calls) >= 4  # 3 chunks + cut(s) + fine
