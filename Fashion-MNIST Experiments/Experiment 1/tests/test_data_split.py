from src.data import split_indices
def test_fixed_split():
    a,b=split_indices(seed=42);c,d=split_indices(seed=42);assert (len(a),len(b))==(50000,10000) and a==c and b==d and not set(a)&set(b)
