
def append(d1, d2) -> dict:
    for k in d2.keys():
        d1[k] = d2[k] + d1.get(k, 0)
    return d1