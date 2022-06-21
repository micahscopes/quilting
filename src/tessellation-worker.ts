// @ts-ignore
import { expose } from "threads/worker"
import { tessellationMesh } from './tessellation';

// create a worker and register public functions
expose({
  tessellationMesh: tessellationMesh,
});
