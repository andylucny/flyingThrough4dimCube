import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

def generate_edges(N):
    # Generate all vertices of the cube (coordinates 0 or 1)
    vertices = list(product([0, 1], repeat=N))

    # Generate all pairs of vertices
    edges = []
    for v1, v2 in combinations(vertices, 2):
        # An edge exists if the two points differ in exactly one coordinate
        if sum(a != b for a, b in zip(v1, v2)) == 1:
            edges.append((np.array(v1,np.float32), np.array(v2,np.float32)))

    return edges

def gram_schmidt(vectors):
    orthonormal_basis = []
    
    for v in vectors:
        # Start with the original vector
        w = np.array(v, dtype=float)
        
        # Subtract the projection onto each previous basis vector
        for u in orthonormal_basis:
            proj = np.dot(w, u) * u
            w = w - proj
        
        # Normalize the vector if it's not zero
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            u = w / norm
            orthonormal_basis.append(u)
    
    return np.array(orthonormal_basis)
    
def orthonormal_basis_orthogonal_to(n): # n is normalized
    m = len(n)
    # Start with identity matrix
    basis = np.eye(m)
    # Subtract projection onto n
    for i in range(m):
        basis[:, i] -= np.dot(basis[:, i], n) * n
    return gram_schmidt(basis)

def intersection(n,d,A,B,base):
    u = B - A
    u = u / np.linalg.norm(u)
    p = n @ u
    if np.abs(p) < 1e-5:
        return [ 
            [A@b for b in base], 
            [B@b for b in base] 
        ]
    t = (d - A @ n) / p
    if t < 0 or t > 1:
        return []
    V = A + t * u 
    z = d * n
    v = V - z
    return [ 
        [v@b for b in base] 
    ]

def same_facet(edge1, edge2):
    A1, B1 = edge1
    A2, B2 = edge2
    for i in range(len(A1)):
        if A1[i] == B1[i] == A2[i] == B2[i]:
            return True
    return False

N = 4
edges = generate_edges(N)

n = [1]*N
n = np.array(n) / np.linalg.norm(n)

base = orthonormal_basis_orthogonal_to(n)

for d in np.linspace(0,np.sqrt(N),10):

    points = []
    for A, B in edges:
        points += [ (A,B,q) for q in intersection(n,d,A,B,base) ]

    segments = []
    for p1, p2 in combinations(points, 2):
        A1, B1, q1 = p1
        A2, B2, q2 = p2
        if same_facet((A1,B1),(A2,B2)):
            segments += [ (q1, q2) ]

    if N == 3:
        # Create a plot
        plt.figure(figsize=(6, 6))

        # Loop through each segment and plot it
        for (x1, y1), (x2, y2) in segments:
            plt.plot([x1, x2], [y1, y2], marker='o')

        # Optional: add grid and equal aspect ratio
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("2D Segments")
        plt.xlabel("X")
        plt.ylabel("Y")

        # Show the plot
        plt.show()

    elif N == 4:
        # Set up the 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each segment
        for (x1, y1, z1), (x2, y2, z2) in segments:
            ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o')

        # Optional: set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Line Segments')

        # Equal aspect ratio (rough approximation)
        ax.set_box_aspect([1, 1, 1])

        plt.show()
