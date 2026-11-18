
# 1. Complement of a Fuzzy Set

A = {"a": 0.2, "b": 0.3, "c": 0.6, "d": 0.6}

Y = {}

print("The Fuzzy Set is:", A)

for key in A:
    Y[key] = 1 - A[key]

print("Fuzzy Set Complement is:", Y)


# 2. Difference Between Two Fuzzy Sets

A = {"a": 0.2, "b": 0.3, "c": 0.6, "d": 0.6}
B = {"a": 0.9, "b": 0.9, "c": 0.4, "d": 0.5}

Y = {}

print("\nThe First Fuzzy Set is:", A)
print("The Second Fuzzy Set is:", B)

for key in A:
    Y[key] = min(A[key], 1 - B[key])

print("Fuzzy Set Difference is:", Y)


# 3. Intersection of Two Fuzzy Sets

Y = {}

for key in A:
    Y[key] = min(A[key], B[key])

print("\nFuzzy Set Intersection is:", Y)


# 4. Union of Two Fuzzy Sets

Y = {}

for key in A:
    Y[key] = max(A[key], B[key])

print("Fuzzy Set Union is:", Y)
