# Improvements

The orientation estimation has improved since the last round of testing because of two steps of the pipeline that have new, improved methods: reconstitution and handedness selection.

The methods used for these two steps are controlled through the ``reconstitution_method`` and ``handedness_method`` parameters of ``build_relative_orientations_graph``.
Currently, the best-performing methods seem to be ``reconstitution_method="soft_voting"`` and ``handedness_method="synchronization"``.
So, it is recommended to call ``build_relative_orientations_graph`` as follows:

```python
from common_lines import build_relative_orientations_graph

graph = build_relative_orientations_graph(
    images,
    reconstitution_method="soft_voting",
    handedness_method="synchronization",
)
```

The parameter ``reconstitution_method`` can also be set to ``"voting"`` with similar results. Its optional ``n_third`` parameter defaults to ``1`` and must satisfy ``1 <= n_third <= n_images - 2``. With ``n_third=1``, ``"voting"`` is equivalent to ``"averaging"``. Use a larger value to combine multiple estimates, for example ``n_third=24`` when at least 26 images are available. ``"soft_voting"`` considers all third images and ignores ``n_third``.

```python
graph = build_relative_orientations_graph(
    images,
    reconstitution_method="voting",
    n_third=24,
    handedness_method="synchronization",
)
```

## Changes

- Introduced ``"voting"`` method for reconstitution (commit f853f59227f0430c325f09e73011f6595ae43b61).
- Introduced ``"synchronization"`` method for handedness selection (commit 376822a638802f3d8f8318c42de7ca1454a95d02).
- Introduced ``"soft_voting"`` method for reconstitution (commit c56248c9dafd7a7bbdb2c7094fa433ebcc6dfcd0).
