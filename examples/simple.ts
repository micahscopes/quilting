import {createPatchProgram} from '../src/patch.js'
import {PicoGL} from 'picogl';

const canvas = document.createElement('canvas');
document.body.appendChild(canvas);
const app = PicoGL.createApp(canvas)
const patchProgram = createPatchProgram(app)
console.log(patchProgram)
