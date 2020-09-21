def palindrome(s:str) -> bool:
	slen = len(s)
	for i in range(slen//2):
		if s[i] != s[slen-1-i]:
			return False
	return True

def longestPalindrome(s: str) -> str:
	lpalstr = ""
	slen = len(s)
	for i in range(slen):
		for j in range(i+1, slen+1):
			subs = s[i:j]
			if palindrome(subs):
				if len(lpalstr) < len(subs):
					lpalstr = subs
	return lpalstr



print(longestPalindrome("aba 2aba"))
