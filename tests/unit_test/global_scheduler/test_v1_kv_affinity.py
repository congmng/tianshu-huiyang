from llumnix.backends.vllm.v1_kv import KVCacheAffinityIndex


class BlockStored:
    def __init__(self, block_hashes, token_ids, block_size, medium="GPU"):
        self.block_hashes = block_hashes
        self.token_ids = token_ids
        self.block_size = block_size
        self.medium = medium


class BlockRemoved:
    def __init__(self, block_hashes):
        self.block_hashes = block_hashes


class AllBlocksCleared:
    pass


def test_v1_kv_affinity_tracks_store_remove_and_clear():
    index = KVCacheAffinityIndex()
    index.apply("a", [BlockStored([1, 2], [10, 11], 2)])
    index.apply("b", [BlockStored([2], [10, 11], 2)])
    assert index.affinity("a", [1, 2, 3]) == 2 / 3
    assert index.rank([2], ["a", "b"]) == ["a", "b"]
    index.apply("a", [BlockRemoved([1])])
    assert index.block_hashes("a") == frozenset({2})
    index.apply("a", [AllBlocksCleared()])
    assert index.block_hashes("a") == frozenset()
