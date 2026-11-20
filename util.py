from itertools import chain, combinations

# function for computing powersets excluding the empty set
def nonempty_powerset(input_set):
    s = list(input_set)
    return [set(combo) for combo in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))][1:]
