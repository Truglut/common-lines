# Handedness selection

## Handedness ambiguity in common lines

Common lines cannot uniquely determine the relative (or absolute) orientations of a set of images: they can only do so up to a "handedness" ambiguity.
Mathematically, if $\{ R_i \}_{i = 1}^n$ is a set of rotations for the images, the data cannot distinguish between the sets
$$ \{ R_i \} \quad \text{and} \quad \{ J R_i J \}, $$
where $J = \operatorname{diag}(1, 1, -1)$.

This is also true for relative rotations: given a pair of images $i, j$, the data cannot distinguish between a rotation $R_{ij}$ and its conjugate $J R_{ij} J$.
For the relative rotations graph, this does not mean two possibilities for each edge, but two global solutions that are not distinguishable: $\{ R_{ij} \}_{i, j = 1}^n$ and $\{ J R_{ij} J \}_{i, j = 1}^n$.

## Choosing a consistent handedness

When applied on an image pair $i, j$, all reconstitution methods will return a matrix $Q_{ij}$ which is (an approximate estimate of) one of $R_{ij}$ or its conjugate $J R_{ij} J$.
Handedness selection is tasked with choosing, for each image pair $i, j$, one of $Q_{ij}$ or $J Q_{ij} J$ so that the resulting graph is internally consistent.
That is, the graph formed by the chosen rotations should approximate either $\{ R_{ij} \}$ or $\{ J R_{ij} J \}$, but not a mixture of both.

There are two implemented methods for this handedness selection step: ``"star"`` and ``"synchronization"``.

### Star handedness selection

``"star"`` attempts to fix a consistent "star" around a given image. Write $R_{ij}$ for the transformation from image $i$ coordinates to image $j$ coordinates. Given a reference pair $(i, j)$, this method fixes $R_{ij} = Q_{ij}$, and for each $k \not \in \{i, j\}$, choose

$$
    R_{ik} = \underset{R \in \{Q_{ik}, J Q_{ik} J\}}{\operatorname{argmin}} \quad \min_{\mu \in \{0, 1 \}} \Vert J^{\mu} Q_{jk} J^{\mu} R_{ij} - R \Vert_F,
$$
where $\Vert \cdot \Vert_F$ denotes the Frobenius norm for matrices.

Then for every edge $k, \ell$, with $k \neq i$ choose the rotation that minimizes the distance to the predicted rotation $R_{i \ell} R_{ik}^T$:

$$
    R_{k \ell} = \underset{R \in \{Q_{k \ell}, J Q_{k \ell} J\}}{\operatorname{argmin}} \Vert R_{i \ell} R_{ik}^T - R \Vert_F.
$$

### J-synchronization handedness selection

The ``"synchronization"`` handedness selection method implements the $J$-synchronization algorithm from:

Pragier, G., Greenberg, I., Cheng, X., & Shkolnisky, Y. (2016). A Graph Partitioning Approach to Simultaneous Angular Reconstitution. IEEE transactions on computational imaging, 2(3), 323–334. <https://doi.org/10.1109/TCI.2016.2557076>. (Section II)

## Note

The only goal of the handedness selection step is to promote internal consistency, and not choose the option that is more likely to be correct (the latter is impossible to do, since both solutions are equally compatible with the data). Also, this approximate handedness consistency is not equivalent to cycle or inverse consistency, which are not enforced in general.
