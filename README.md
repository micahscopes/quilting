# quilting
work in progress

### todo
- [x] proof of concept with CPU
- [x] proof of concept with GPU
- [x] compute normals
  - [x] method (1): tried automatic differentiation with `Cl{4,1} x Cl{0,0,2}` and failed
    - complete overkill
  - [x] method (2): tried with `Cl{4,1,2}` and the shader actually compiled but I didn't figure out how to get the normals
  - [x] method (3): computed nearby points at `u+eps, v+eps` and used the cross product, seems to work! it's fast too.
  - [ ] method (4): by finding the intrinsic Moebius transformation of a patch and using it as described in Karpavicius and Krasauskas paper
    - this might result in a performance gain of 2-3x
- [x] triangular patches
- [-] multiple patches per shader (?)
  - [x] try batch mode
    - getting decent performance
  - [ ] try instancing
  - [ ] try one shader per precomputed LOD tessellation + batching/instancing
- [-] convert polygon mesh to patches
 - [x] prototype handling triangular mesh
 - [ ] something with a nice interface
- [ ] control point/weight CGA transformations
  - [ ] first try on CPU
  - [ ] try on GPU in main vertex shader
- [ ] initial triangulation of UV square with variable level of detail
  - [x] deterministic constraint at edges
  - [x] poisson sampling with RTree index for uniform distribution of inner points
    - [no] add and remove points continuously?
  - [x] using delaunay triangulation of sampled points
- [ ] compute LOD estimator
  - using the method described in _Real-time visualization of Moebius transformations in space using Quaternionic-Bezier approach_ by Karpavicius and Krasauskas
- [ ] pre-compute transformations + LOD + normals in fragment shader pass
  - [ ] switch to pex-context?
- [ ] rework API
- [ ] tests
- [ ] examples
- [ ] documentation
