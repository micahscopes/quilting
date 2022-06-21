import barycentric from "barycentric";
import { flatten, fromPairs, sortBy, uniqBy, uniqWith } from "lodash-es";
import { permutationsWithReplacement } from "combinatorial-generators";
import cumulativeSum from "cumulative-sum";
import { permutationIndices3, vertPermFromEdgePerm } from "./permutator";

import { spawn, Pool, Worker } from "threads";
const pool = Pool(() =>
  spawn(
    new Worker(
      new URL("./tessellation-worker.ts", import.meta.url),
      { type: "module" }
    )
  )
); //, 8 /* optional size */)

export const loadTessellationAtlas = async (LODs: number[]) => {
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

  console.log(reducedLODs);
  // const meshes = reducedLODs.map()
  // const meshes = reducedLODs.map(k => allTessellations[JSON.stringify(k)])
  // console.log(meshes);
  //
  const meshes = await Promise.all(
    reducedLODs.map((lods) => pool.queue(({tessellationMesh}) => tessellationMesh(lods)))
  );

  // const meshes = reducedLODs.map((sideLODs) => tessellationMesh(sideLODs as triResolutions)
  // );

  // const combinedMesh = meshCombine(meshes);
  const combinedMesh = meshes.reduce(
    (acc, next) => {
      console.log(acc, next);
      // next.cellPositions = next.cells.map(cell => cell.map(i => next.positions[i]));
      return {
        cells: [...acc.cells, ...next.cells],
        positions: [
          ...acc.positions,
          ...flatten(next.cellPositions).map((xyz: any) =>
            barycentric(
              [
                [0, 0],
                [1, 0],
                [0, 1],
              ],
              xyz.slice(0, 2)
            )
              .map((x: number) => x * 2 ** 15)
              .map(Math.floor)
          ),
        ],
      };
    },
    { cells: [], positions: [] }
  );

  const counts = reducedLODs.map((_, i) => meshes[i].cellPositions.length * 3);
  const baseIndices = [0, ...cumulativeSum(counts.map((x) => x)).slice(0, -1)];
  const lookup = fromPairs(
    uniqWith(
      reducedLODs.flatMap((lod, i) =>
        permutationIndices3.map((pidxs) => [
          JSON.stringify(pidxs.map((pi: any) => lod[pi])),
          i,
          lod,
          pidxs,
        ])
      ),
      ([a], [b]) => a === b
    ).map(([prm_i, lod_i, lod, edgePermutation]) => [
      prm_i,
      {
        lod,
        baseIndex: baseIndices[lod_i],
        count: counts[lod_i],
        edgePermutation,
        permutation: vertPermFromEdgePerm(edgePermutation),
      },
    ])
  );
  return { meshes, combinedMesh, counts, baseIndices, lookup };
};
