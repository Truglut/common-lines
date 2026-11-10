# Reconstitution

Reconstitution is the process of estimating the relative orientation of images from common lines.
The common line between two images is not enough to determine their relative orientation, so for each image pair it is necessary to select at least one additional image for the reconstitution process.
Each additional image with valid common-line geometry provides an estimate of the pair’s relative orientation, so estimates from multiple third images must be aggregated.

There are currently three different reconstitution methods that differ precisely in how they aggregate said estimates:

- ``"averaging"`` aligns the handedness branches of the third-image rotation estimates, averages the matrices, and projects the average onto $SO(3)$.
- ``"voting"`` uses the fact that, when the rotations are parameterized in a ZYZ Euler convention, the common line between two images determines two out of three Euler angles of their relative rotation, while every possible third image gives an estimate for the third Euler angle. With this method, every third image casts a Gaussian vote in the $[0, \pi]$ interval to form a "voting histogram", and the final angle is the maximum of said histogram.
- ``"soft_voting"`` uses the same procedure as ``"voting"``, but it adds a reliability score to each common line that weighs its contribution to the vote. The reliabilities are re-calculated after voting, so the method is iterative.

Pairs without usable third-image support receive a canonical in-plane fallback rotation.
