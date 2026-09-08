"""
Cache storage for World Model hidden states.

TODO:
- implement token indexing
- implement frame-wise cache update
- connect with Wan2.1 blocks
"""


class CacheManager:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key, None)

    def update(self, key, value):
        self.cache[key] = value

    def clear(self):
        self.cache = {}
