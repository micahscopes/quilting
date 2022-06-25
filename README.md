# quilting
#### Tessellation and parametric surface tools for WebGL2
[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
##### Examples are licensed for non-commercial use at this time:
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


![Screenshot from 2022-06-24 13-42-42](https://user-images.githubusercontent.com/389782/175647917-effa3246-f015-44e8-b2e6-670785b8a47f.png)


<!-- ![Screenshot from 2022-04-16 22-06-19](https://user-images.githubusercontent.com/389782/169445957-f268cf18-0881-4ec7-b283-c68c836f7369.png) -->


<!-- ## coming soon...
![unknown-9](https://user-images.githubusercontent.com/389782/169445496-c760ec7a-4f7f-4bc2-a640-bed6be4e8e5a.png)


https://user-images.githubusercontent.com/389782/169445431-d8bdbf5a-977a-46a0-98d1-b2a801fda0f7.mp4 -->



### todo

- [x] tessellation atlas generator
  - [x] using web workers
  - [x] symmetry-aware using permutation indices on control mesh patches
  - [x] try pre-computed tessellation meshes
    - tried this and found that the meshes were just too big... using workers to generate tessellations is faster and more bandwidth-efficient
- [x] basic tessellation example
  - [ ] fallback support for browsers without WebGL2 draft extensions
  - [ ] mobile support
- [x] simple "terrain" example
- [ ] quaternion Bezier surface patch example
  - [ ] compute LODs + control point transformations on GPU
    - [x] proof of concept
  - [ ] draggable control points and weights
  - [ ] manual moebius transformation input w/normalization
  - [ ] draw both original and transformed patches
  - [ ] basic matcap shading
- [ ] tessellation atlas optimizations
  - [ ] faster generation for high LODs
  - [ ] better compression
- [ ] blog post
- [ ] matcap-based spherical reflection example
  - [ ] draggable reflection sphere control widget
  - [ ] glTF scene geometry loader
  - [ ] scene graph integrating general Moebius transformations
- [ ] refactor/modularize
  - [ ] documentation
- [ ] more fully featured glTF loader

---
https://user-images.githubusercontent.com/389782/175750591-c230b927-ac74-46e8-ac75-eb05f7a7042b.mp4

