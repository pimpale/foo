def lengthOfLongestSubstring(s: str) -> int:
	base = 0
	maxlen = 0
	while True:
		chars = set()
		for i, c in enumerate(s[base:]):
			print("chars:", chars)
			print("char:", c)

			if c in chars:
				break

			chars.add(c)

		base = base + len(chars)
		maxlen = max(maxlen, len(chars))

		if len(s[base:]) <= maxlen:
			break

	return maxlen

print(lengthOfLongestSubstring("dvdf"))
