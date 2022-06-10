import { makeTessellationAtlas } from '../src/tessellation';
import meshCombine from 'mesh-combine';

const atlas = makeTessellationAtlas([0,2].map(x => 2**(2*x)))
console.log('meshes', atlas)