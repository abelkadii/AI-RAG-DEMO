from models import EvidenceChunk
from vector_store import HashEmbedder, PersistentVectorStore


def test_persistent_retrieval_preserves_metadata(tmp_path):
    store = PersistentVectorStore(tmp_path, HashEmbedder(128))
    store.build([EvidenceChunk(chunk_id="c1", page=17, section="Reliability", text="prepare for component failures")])
    loaded = PersistentVectorStore(tmp_path, HashEmbedder(128))
    result = loaded.search("component failures", 1)[0]
    assert result.page == 17
    assert result.section == "Reliability"

