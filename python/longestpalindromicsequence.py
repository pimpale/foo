def isOddShellPal(s:str, slen:int, i:int, r:int) -> bool:
    li = i-r
    hi = i+r

    if li < 0 or hi >= slen:
        return False
    return s[li] == s[hi]

# ibefore is the index of the element before the point we want to check
def isEvenShellPal(s:str, slen:int, ibefore:int, r:int) -> bool:
    li = ibefore-r+1
    hi = ibefore+r

    if li < 0 or hi >= slen:
        return False
    return s[li] == s[hi]

def getOddPal(s:str, i:int, r:int) -> str:
    li = i-r
    hi = i+r
    return s[li:hi+1]


# ibefore is the index of the element before the point we want to check
def getEvenPal(s:str, ibefore:int, r:int) -> str:
    li = ibefore-r+1
    hi = ibefore+r
    return s[li:hi+1]

def longestPalindrome(s: str) -> str:
    bigpal = ""
    bigpallen = 0

    slen = len(s)

    # odd check
    for i in range(slen):
        r = 0

        while True:
            if not isOddShellPal(s,slen, i, r+1):
                break
            r += 1

        pallen = r*2 +1
        if pallen > bigpallen:
            bigpal = getOddPal(s, i, r)
            bigpallen = pallen

    # even check
    for i in range(slen):
        r = 0

        while True:
            if not isEvenShellPal(s,slen, i, r+1):
                break
            r += 1

        pallen = r*2
        if pallen > bigpallen:
            bigpal = getEvenPal(s, i, r)
            bigpallen = pallen

    return bigpal


print(longestPalindrome("abc baab"))
