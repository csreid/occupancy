from collections import OrderedDict

class LRUCache:
	def __init__(self, max_size=100):
		self.cache = OrderedDict()
		self.max_size = max_size

	def get(self, key):
		if key not in self.cache:
			return None

		value = self.cache.pop(key)
		self.cache[key] = value
		return value

	def put(self, key, value):
		if key in self.cache:
			self.cache.pop(key)
		elif len(self.cache) >= self.max_size:
			self.cache.popitem(last=False)

		self.cache[key] = value

	def clear(self):
		self.cache.clear()

	def __len__(self):
		return len(self.cache)
