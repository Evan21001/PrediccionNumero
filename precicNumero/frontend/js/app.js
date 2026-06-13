const drawingBoard   = document.getElementById('canvas');
const renderCtx      = drawingBoard.getContext('2d');
const resetButton    = document.getElementById('clear-btn');
const analyzeButton  = document.getElementById('predict-btn');
const outputDisplay  = document.getElementById('result');

let activeStroke    = false;
let digitClassifier = null;

// Rellena la superficie con blanco para que el modelo reciba fondo limpio
renderCtx.fillStyle   = 'white';
renderCtx.fillRect(0, 0, drawingBoard.width, drawingBoard.height);
renderCtx.lineWidth   = 14;
renderCtx.lineCap     = 'round';
renderCtx.lineJoin    = 'round';
renderCtx.strokeStyle = 'black';

async function initializeClassifier() {
    outputDisplay.innerText = 'Cargando IA...';
    try {
        digitClassifier = await tf.loadLayersModel('./modelo_web/model.json');
        outputDisplay.innerText = '—';
        console.log('Clasificador listo.');
    } catch (err) {
        console.error('No se pudo cargar el modelo TF.js:', err);
        outputDisplay.innerText = 'Error IA';
    }
}
initializeClassifier();

// Registro de eventos de ratón para captura del trazo
drawingBoard.addEventListener('mousedown',  (e) => { activeStroke = true; handleStroke(e); });
drawingBoard.addEventListener('mousemove',  handleStroke);
drawingBoard.addEventListener('mouseup',    () => { activeStroke = false; renderCtx.beginPath(); });
drawingBoard.addEventListener('mouseleave', () => { activeStroke = false; renderCtx.beginPath(); });

function handleStroke(evt) {
    if (!activeStroke) return;
    const bounds  = drawingBoard.getBoundingClientRect();
    const cursorX = evt.clientX - bounds.left;
    const cursorY = evt.clientY - bounds.top;

    renderCtx.lineTo(cursorX, cursorY);
    renderCtx.stroke();
    renderCtx.beginPath();
    renderCtx.moveTo(cursorX, cursorY);
}

resetButton.addEventListener('click', () => {
    renderCtx.fillStyle = 'white';
    renderCtx.fillRect(0, 0, drawingBoard.width, drawingBoard.height);
    outputDisplay.innerText = '—';
});

/**
 * Recorta el trazo usando su bounding box, lo centra en un canvas cuadrado
 * y lo escala a 28×28 — el formato de entrada que espera el modelo MNIST.
 */
function normalizeDrawing(sourceCanvas) {
    const srcCtx  = sourceCanvas.getContext('2d');
    const imgData = srcCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    const pixels  = imgData.data;

    let left   = sourceCanvas.width;
    let right  = 0;
    let top    = sourceCanvas.height;
    let bottom = 0;
    let hasContent = false;

    for (let row = 0; row < sourceCanvas.height; row++) {
        for (let col = 0; col < sourceCanvas.width; col++) {
            const idx = (row * sourceCanvas.width + col) * 4;
            if (pixels[idx] < 220 && pixels[idx + 1] < 220 && pixels[idx + 2] < 220) {
                if (col < left)   left   = col;
                if (col > right)  right  = col;
                if (row < top)    top    = row;
                if (row > bottom) bottom = row;
                hasContent = true;
            }
        }
    }

    if (!hasContent) return null;

    // Padding de 40px para evitar que el dígito toque los bordes del recorte
    const pad = 40;
    left   = Math.max(0, left   - pad);
    right  = Math.min(sourceCanvas.width,  right  + pad);
    top    = Math.max(0, top    - pad);
    bottom = Math.min(sourceCanvas.height, bottom + pad);

    const cropW      = right  - left;
    const cropH      = bottom - top;
    const squareSide = Math.max(cropW, cropH);

    const squareCanvas    = document.createElement('canvas');
    squareCanvas.width    = squareSide;
    squareCanvas.height   = squareSide;
    const squareCtx       = squareCanvas.getContext('2d');

    squareCtx.fillStyle = 'white';
    squareCtx.fillRect(0, 0, squareSide, squareSide);

    // Centrado simétrico del recorte dentro del canvas cuadrado
    const offsetX = (squareSide - cropW) / 2;
    const offsetY = (squareSide - cropH) / 2;
    squareCtx.drawImage(sourceCanvas, left, top, cropW, cropH, offsetX, offsetY, cropW, cropH);

    const canvas28    = document.createElement('canvas');
    canvas28.width    = 28;
    canvas28.height   = 28;
    const ctx28       = canvas28.getContext('2d');
    ctx28.imageSmoothingEnabled = true;
    ctx28.imageSmoothingQuality = 'high';
    ctx28.drawImage(squareCanvas, 0, 0, squareSide, squareSide, 0, 0, 28, 28);

    return canvas28;
}

analyzeButton.addEventListener('click', () => {
    if (!digitClassifier) {
        alert('El modelo web aún se está cargando. Espera un segundo.');
        return;
    }

    const processedCanvas = normalizeDrawing(drawingBoard);
    if (!processedCanvas) {
        outputDisplay.innerText = '—';
        return;
    }

    tf.tidy(() => {
        // Convierte a escala de grises, invierte colores (MNIST: fondo negro, trazo blanco)
        let inputTensor = tf.browser.fromPixels(processedCanvas);
        inputTensor = inputTensor.mean(2).expandDims(-1);
        inputTensor = tf.scalar(1.0).sub(inputTensor.div(tf.scalar(255.0)));

        // Suprime ruido sub-umbral menor al 5% de intensidad
        const noiseFloor = tf.scalar(0.05);
        inputTensor = tf.where(inputTensor.greater(noiseFloor), inputTensor, tf.zerosLike(inputTensor));

        // Agrega dimensión de lote → [1, 28, 28, 1]
        inputTensor = inputTensor.expandDims(0);

        const scores      = digitClassifier.predict(inputTensor);
        const prediction  = scores.argMax(1).dataSync()[0];
        const allScores   = scores.dataSync();
        const confidence  = (allScores[prediction] * 100).toFixed(1);

        console.log(`Predicción: ${prediction} | Confianza: ${confidence}%`);
        outputDisplay.innerText = prediction;
    });
});
