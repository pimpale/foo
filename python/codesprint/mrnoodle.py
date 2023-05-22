import sys
import numpy as np

def theta(L, r):
    return 2 * np.arcsin(0.5 * L / r)

def find_radius(lengths, epsilon):
    min_theta = 1 - epsilon
    max_theta = 1 + epsilon

    min_radius = np.max(lengths)/2
    max_radius = np.sum(lengths)/2

    while True:
        radius = (0.5 * min_radius) + (0.5 * max_radius);
        sum_radius = np.sum(theta(lengths, radius))/(2*np.pi)

        if sum_radius >= min_theta and sum_radius <= max_theta:
            break
        elif sum_radius < 1:
            max_radius = radius
        else:
            min_radius = radius

    return radius

def get_points(radius, lengths):
    points = np.zeros((len(lengths), 2))
    phi = -0.5 * theta(lengths[0], radius)

    points[0, 0] = radius * np.cos(phi)
    points[0, 1] = -radius * np.sin(phi)

    for i in range(1, len(lengths)):
        points[i, 0] = points[i-1, 0] - radius * np.cos(phi)
        points[i, 1] = points[i-1, 1] + radius * np.sin(phi)
        phi += theta(lengths[i], radius)

    return points



# algorithm welzl is[8]
#     input: Finite sets P and R of points in the plane |R| ≤ 3.
#     output: Minimal disk enclosing P with R on the boundary.
# 
#     if P is empty or |R| = 3 then
#         return trivial(R)
#     choose p in P (randomly and uniformly)
#     D := welzl(P − {p}, R)
#     if p is in D then
#         return D
# 
#     return welzl(P − {p}, R ∪ {p})

def wezl(points, boundary):
    if len(points) == 0 or len(boundary) == 3:
        return trivial_wezl(boundary)
    p = points[0]
    center, radius = wezl(points[1:], boundary)
    if np.linalg.norm(p - center) <= radius:
        return center, radius

    return wezl(points[1:], np.vstack([boundary, p]))

def trivial_wezl(boundary:np.ndarray):
    if len(boundary) == 0:
        return np.zeros(2), 0
    elif len(boundary) == 1:
        return boundary[0], 0
    elif len(boundary) == 2:
        center = (boundary[0] + boundary[1])/2
        radius = np.linalg.norm(boundary[0] - center)
        return center, radius
    else:
        ax, ay = boundary[0]
        bx, by = boundary[1]
        cx, cy = boundary[2]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
        r = np.linalg.norm(boundary[0] - (ux, uy))
        return np.array([ux, uy]), r

def main():
    length_strs = sys.stdin.readlines()[1].split()
    lengths = np.array([int(l) for l in length_strs])


    circumscribed_radius = find_radius(lengths, 1e-8)
    print(circumscribed_radius)


main()