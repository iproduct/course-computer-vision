import numpy as np

# Define our two vectors from the numerical example
a = np.array([2, 3, 4])
b = np.array([5, 6, 7])

# Compute the 3D cross product
c = np.cross(a, b)

print("Cross Product Vector c:", c)
# Output: [ -3  6 -3 ]

# Optional: Sanity check using the dot product (should equal 0)
dot_a = np.dot(a, c)
dot_b = np.dot(b, c)

print(f"Dot product with a: {dot_a} | Dot product with b: {dot_b}")
# Output: Dot product with a: 0 | Dot product with b: 0
