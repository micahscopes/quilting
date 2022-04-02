import {createPatchPrograms} from '../src/patch.js'
import {PicoGL} from 'picogl';

const canvas = document.createElement('canvas');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
document.body.appendChild(canvas);
const app = PicoGL.createApp(canvas)
.clearColor(0.0, 0.0, 0.0, 1.0);


// GEO DATA
const NUM_INSTANCES = 10;

let offsetData = new Float32Array(NUM_INSTANCES * 2);
let rotationData = new Float32Array(NUM_INSTANCES);
let colorData = new Uint8Array(NUM_INSTANCES * 3);
let positionData = new Float32Array([
    0.012, 0.0,
    -0.008, 0.008,
    -0.008, -0.008,
]);

for (let i = 0; i < NUM_INSTANCES; ++i) {
    let oi = i * 2;
    let ri = i;
    let ci = i * 3;

    offsetData[oi] = Math.random() * 2.0 - 1.0;
    offsetData[oi + 1] = Math.random() * 2.0 - 1.0;

    rotationData[i] = Math.random() * 2 * Math.PI;

    colorData[ci]     = Math.floor(Math.random() * 256);
    colorData[ci + 1] = Math.floor(Math.random() * 256);
    colorData[ci + 2] = Math.floor(Math.random() * 256);
}

// INPUT AND OUTPUT VERTEX BUFFERS
let offsetsA = app.createVertexBuffer(PicoGL.FLOAT, 2, offsetData);
let offsetsB = app.createVertexBuffer(PicoGL.FLOAT, 2, offsetData.length);

let rotationsA = app.createVertexBuffer(PicoGL.FLOAT, 1, rotationData);
let rotationsB = app.createVertexBuffer(PicoGL.FLOAT, 1, rotationData.length);


// ATTRIBUTES FOR DRAWING
let positions = app.createVertexBuffer(PicoGL.FLOAT, 2, positionData);
let colors = app.createVertexBuffer(PicoGL.UNSIGNED_BYTE, 3, colorData);


// COMBINE VERTEX BUFFERS INTO INPUT AND OUTPUT VERTEX ARRAYS
let updateArrayA = app.createVertexArray()
.vertexAttributeBuffer(0, offsetsA)
.vertexAttributeBuffer(1, rotationsA);

let updateArrayB = app.createVertexArray()
.vertexAttributeBuffer(0, offsetsB)
.vertexAttributeBuffer(1, rotationsB);

// CREATE TRANSFORM FEEDBACK FROM INPUT AND OUTPUT VERTEX ARRAYS
let transformFeedbackA = app.createTransformFeedback()
.feedbackBuffer(0, offsetsA)
.feedbackBuffer(1, rotationsA);

let transformFeedbackB = app.createTransformFeedback()
.feedbackBuffer(0, offsetsB)
.feedbackBuffer(1, rotationsB);

// VERTEX ARRAYS FOR DRAWING
let drawArrayA = app.createVertexArray()
.vertexAttributeBuffer(0, positions)
.instanceAttributeBuffer(1, colors, { normalized: true })
.instanceAttributeBuffer(2, offsetsA)
.instanceAttributeBuffer(3, rotationsA);

let drawArrayB = app.createVertexArray()
.vertexAttributeBuffer(0, positions)
.instanceAttributeBuffer(1, colors, { normalized: true })
.instanceAttributeBuffer(2, offsetsB)
.instanceAttributeBuffer(3, rotationsB);


const {updateProgram, drawProgram} = createPatchPrograms(app)

    // CREATE DRAW CALLS FOR SIMULATION
    // A BUFFERS AS INPUT, UPDATE B BUFFERS
    let updateDrawCallA = app.createDrawCall(updateProgram, updateArrayA)
    .primitive(PicoGL.POINTS)
    .transformFeedback(transformFeedbackB);

    // B BUFFERS AS INPUT, UPDATE A BUFFERS
    let updateDrawCallB = app.createDrawCall(updateProgram, updateArrayB)
    .primitive(PicoGL.POINTS)
    .transformFeedback(transformFeedbackA);

    // DRAW USING CONTENTS OF A BUFFERS
    let drawCallA = app.createDrawCall(drawProgram, drawArrayA);

    // DRAW USING CONTENTS OF B BUFFERS
    let drawCallB = app.createDrawCall(drawProgram, drawArrayB);

    // START BY DRAWING B BUFFERS THAT ARE
    // WRITTEN TO ON FIRST TRANSFORM FEEDBACK
    // PASS
    let updateDrawCall = updateDrawCallA;
    let mainDrawCall = drawCallB;
    function draw() {
        // if (timer.ready()) {
        //     utils.updateTimerElement(timer.cpuTime, timer.gpuTime);
        // }

        // timer.start();

        // TRANSFORM FEEDBACK
        app.enable(PicoGL.RASTERIZER_DISCARD)
        updateDrawCall.draw()

        // DRAW
        app.disable(PicoGL.RASTERIZER_DISCARD).clear()
        mainDrawCall.draw();

        // SWAP INPUT AND OUTPUT BUFFERS
        updateDrawCall = updateDrawCall === updateDrawCallA ? updateDrawCallB : updateDrawCallA;
        mainDrawCall = mainDrawCall === drawCallA ? drawCallB : drawCallA;

        // timer.end();

        requestAnimationFrame(draw);
    }

    requestAnimationFrame(draw);
