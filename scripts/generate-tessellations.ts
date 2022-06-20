import {
  flatten,
  pickBy,
  sortBy,
  uniqBy,
  fromPairs,
  uniqWith,
} from "lodash-es";
import {
  tessellationMesh,
  triPatch,
  triResolutions,
} from "../src/tessellation";

import { writeFile } from "fs/promises";
import { existsSync, createWriteStream } from "fs";

// import tessellations from "../tessellations/*.json";
const byteSize = (str: string) => new Blob([str]).size;

import { unpack, pack, PackrStream } from "msgpackr";

// const meshlet = tessellationMesh(
//   [10, 10, 10].map((x: number) => 2 ** x) as triResolutions
// );
// console.log(
//   byteSize(
//     JSON.stringify({
//       positions: meshlet.positions.map((cell: any) =>
//         cell.map((position: any) => Math.ceil(position * 2 ** 16))
//       ),
//       cells: meshlet.cells,
//       //   cellPositions: meshlet.cellPositions.map((cell: any) =>
//       //     cell.map((corner: any) =>
//       //       corner.map((xyz: any) => Math.ceil(xyz * 2 ** 16))
//       //     )
//       //   ),
//     })
//   ) /
//     1024 /
//     1024
// );

import { permutationsWithReplacement } from "combinatorial-generators";
import cumulativeSum from "cumulative-sum";
import { permutationIndices3, vertPermFromEdgePerm } from "../src/permutator";

// // @ts-ignore
// import meshCombine from "mesh-combine";
// // @ts-ignore
// import barycentric from "barycentric";

export const generateTessellations = (LODs: number[]) => {
  console.log();
  const reducedLODs = uniqBy(
    [
      ...permutationsWithReplacement(
        LODs.sort((a, b) => b - a),
        3
      ),
    ],
    (x) => JSON.stringify(sortBy(x))
  );

  reducedLODs.forEach((sideLODs) => {
    if (existsSync(`./tessellations/${sideLODs.join('-')}.bin`)) {
      console.log('tessellation already generated for', sideLODs)
      return
    }
    console.log('generating tessellations for', sideLODs)
    const mesh = tessellationMesh(sideLODs as triResolutions);
    const lodMesh = {
      positions: mesh.positions.map((cell: any) =>
        cell.map((position: any) => Math.ceil(position * 2 ** 12)).slice(0, 2)
      ),
      cells: mesh.cells,
      lod: sideLODs,
      permutations: {} as any,
    };
    uniqWith(
      [sideLODs].flatMap((lod, i) =>
        permutationIndices3.map((pidxs) => [
          JSON.stringify(pidxs.map((pi: any) => lod[pi])),
          i,
          lod,
          pidxs,
        ])
      ),
      ([a], [b]) => a === b
    ).forEach(([prm_i, lod_i, lod, edgePermutation]) => {
      lodMesh.permutations[prm_i] = {
        lod,
        lodKey: JSON.stringify(lod),
        permutationLodKey: prm_i,
        edgePermutation,
        permutation: vertPermFromEdgePerm(edgePermutation),
      };
    });
    const data$ = new PackrStream()
    const file$ = createWriteStream(`./tessellations/${lodMesh.lod.join("-")}.bin`)
    data$.pipe(file$)
    data$.write(lodMesh)
    writeFile(`./tessellations/${lodMesh.lod.join("-")}.json`, JSON.stringify(lodMesh));
    console.log('finished generating for', sideLODs);
  });
};

const lods = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((x) => 2 ** x)
// const lods = [0, 1, 2, 3, 4, 5, 6, 7].map((x) => 2 ** x)
console.log("Generating tessellations for lods:", lods);
generateTessellations(lods) as any;
// console.log(tess.meshes['[2,2,1]'].permutations)

// console.log("encoding...");
// const packedTess = pack(tess);
// writeFile(`./tessellations/${lods.join("-")}`, packedTess);

// console.log(pack(tess).length / 1024 / 1024, "MB");
