# Common Lines Orientation Estimation

Estimate initial relative orientations of cryo-EM particle images using common lines.

The main output is a directed graph of 3D rotation matrices. For each pair of images ``i`` and ``j``,

$$
\text{graph}[i, j] = R_{i \to j}
$$

transforms vector coordinates from the coordinate system of image ``i`` to the coordinate system of image ``j``:

```python
v_j = graph[i, j] @ v_i
```

See [Rotation convention](#rotation-convention) for more details on the precise convention used throughout the code.

## Installation

From the repository root:

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Basic usage

```python
from common_lines import build_relative_orientations_graph

graph = build_relative_orientations_graph(images)
```

images must be a NumPy array with shape ``(n_images, height, width)`` containing square particle images.

The returned array has shape ``(n_images, n_images, 3, 3)``, and each block ``graph[i, j]`` is a 3D rotation matrix describing the estimated relative orientation between images ``i`` and ``j``.

## Rotation convention

Rotations are always interpreted as transformations between coordinate systems/change-of-basis matrices.

If ``v_i`` contains the coordinates of a 3D vector in the coordinate system associated with image ``i``, then

```python
v_j = graph[i, j] @ v_i
```

contains the coordinates of the same vector in the coordinate system associated with image j.
Therefore,

$$
\text{graph}[i, j] = R_{i \to j}
$$

If ``R_i`` and ``R_j`` transform coordinates from some common reference system into the coordinate systems of images ``i`` and ``j``, respectively, then ideally

```python
graph[i, j] = R_j @ R_i.T
```

Pairwise relative orientations are estimated independently. The graph is not explicitly constrained to satisfy exact inverse or cycle consistency.

For each pair, up to `n_third` geometrically valid third images are sampled
without replacement. Their tilt estimates cast Gaussian votes with a five-degree
bandwidth on the interval [0, π]. The largest grid peak is refined continuously
to avoid rounding the estimated tilt to a bin. The tilt and its negative give
the two handedness candidates. With the default `n_third=1`, voting retains the
single triplet estimate; use more third images to obtain a consensus.

The lower-level `estimate_relative_orientation_candidates` function also accepts
`method="averaging"` to average triplet rotations after aligning their handedness
branches. Its default is `method="voting"`.

## Handedness ambiguity

Common-line geometry cannot distinguish between two globally mirrored orientation solutions.
``build_relative_orientations_graph()`` returns one of these two equivalent handedness solutions.

The opposite solution can be obtained with:

```python
from common_lines import mirror_relative_orientations_graph

mirrored_graph = mirror_relative_orientations_graph(graph)
```

The mirrored graph follows exactly the same coordinate convention:

```python
v_j = mirrored_graph[i, j] @ v_i
```

The two solutions differ only by the global handedness ambiguity inherent to common-line angular reconstitution.

## Precentering

Common-line geometry assumes that the particle is centered consistently across images, which is not always true for experimental data.

When precentering is enabled, this implementation estimates a single shift for each image rather than including many possible translations in the common-line search. Two methods are available: center-of-mass and reflection centering. These can be selected by passing either `precenter_method="center_of_mass"` or `precenter_method="reflection"` to `build_relative_orientations_graph`.

Center-of-mass centering is motivated by the fact that, under the ideal projection model, the center of mass of a projection coincides with the projection of the volume's center of mass. The estimated shifts from either method are applied to the original images.

However, neither method has improved results in our experimental-data tests so far, and in some cases precentering appears to reduce accuracy. Precentering is therefore disabled by default (`precenter_method=None`).

## Preprocessing

Preprocessing is used to make experimental images better match the assumptions made by the precentering and common-line detection stages.

In particular, the preprocessing aims to produce images in which:

- the background is approximately zero;
- the particle is the main source of positive signal;
- relative image intensities are preserved as much as possible.

These conditions are especially important for center-of-mass precentering, since background offsets or unrelated positive signal can bias the estimated center.

Preprocessing is disabled by default (`preprocessing_options=None`). It can be enabled explicitly by passing a `PreprocessingOptions` instance to `build_relative_orientations_graph`. An instance created with `PreprocessingOptions()` also leaves the images unchanged.

When precentering is enabled, the pipeline uses the same preprocessing configuration at two stages:

1. It preprocesses copies of the input images to estimate the centering shifts.
2. It applies those shifts to the original images, then preprocesses the shifted images before common-line detection.

When precentering is disabled, preprocessing is applied only once, before common-line detection. Setting `preprocessing_options=None` disables preprocessing at both stages without disabling precentering.

Preprocessing can be customized with `PreprocessingOptions`:

```python
from common_lines import (
    PreprocessingOptions,
    build_relative_orientations_graph,
)

graph = build_relative_orientations_graph(
    images,
    preprocessing_options=PreprocessingOptions(
        clip_negative=True,
    ),
)
```

A preprocessing pipeline that uses an inscribed circular mask, estimates the background from the region outside that mask, subtracts that background, and leaves negative values unchanged can be obtained with

```python
preprocessing_options = PreprocessingOptions.default_from_images(images)
```

The current preprocessing choices have not yet been evaluated extensively on experimental data, and have not shown improved performance on the tests that have been run. If a dataset already has a suitable external preprocessing pipeline, it may be preferable to use that and disable the built-in preprocessing:

```python
graph = build_relative_orientations_graph(
    externally_preprocessed_images,
    preprocessing_options=None,
)
```

## Diagnostics and fallback estimates

The normal API returns only the orientation graph:

```python
graph = build_relative_orientations_graph(images)
```

If diagnostic information is useful, use:

```python
from common_lines import build_relative_orientations_graph_with_diagnostics

result = build_relative_orientations_graph_with_diagnostics(images)

graph = result.graph
used_fallback = result.used_fallback
handedness_reference = result.handedness_reference
```

Normally, a relative orientation is reconstructed using a third image. For some image pairs, no tested third image may produce a valid angular-reconstitution geometry. Rather than failing the complete estimation, the code uses a deterministic pairwise fallback for those edges.

``used_fallback[i, j]`` indicates whether this occurred for the directed edge from image i to image j.

The fallback still respects the same convention:

```text
graph[i, j] : image i coordinates -> image j coordinates
```

but a single common line cannot determine a complete 3D relative orientation. The fallback therefore chooses a canonical in-plane rotation consistent with that common line and should be regarded as an initialization of lower confidence than a full angular-reconstitution estimate.

``result.handedness_reference`` is a tuple containing the indices of the image pair that was used to fix one consistent handedness for the whole graph.

## Typical workflow

For the intended use case, the normal workflow is simply:

```python
from common_lines import build_relative_orientations_graph

graph = build_relative_orientations_graph(images)

# Example: relative orientation from image i to image j
R_ij = graph[i, j]
```

In most cases there should be no need to interact directly with the common-line detection, Radon transform, angular-reconstitution, or centering internals.

## Authorship

Author: Andrés Contreras

Affiliation: National Center for Biotechnology (CNB-CSIC), Madrid, Spain

Contact: <andres.contreras@cnb.csic.es>
