const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

const imageSize = [150, 150];
const batchSize = 32;
const testDataDir = 'dataset/validation';

// Load images from the directory
async function loadImageFromDir(dir) {
    const files = fs.readdirSync(dir);
    const images = [];

    for (let file of files) {
        const ext = path.extname(file).toLowerCase();
        // Skip non-image files and hidden files like .DS_Store
        if (!['.jpg', '.jpeg', '.png', '.bmp', '.gif'].includes(ext) || file.startsWith('.')) {
            continue;
        }
        
        const imgPath = path.join(dir, file);
        
        try {
            // Load the image using canvas
            const img = await loadImage(imgPath);
            const canvas = createCanvas(imageSize[0], imageSize[1]);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, imageSize[0], imageSize[1]);
            const imageData = ctx.getImageData(0, 0, imageSize[0], imageSize[1]);
            const imageTensor = tf.tensor3d(new Uint8Array(imageData.data), [imageSize[1], imageSize[0], 4], 'int32')
                .slice([0, 0, 0], [-1, -1, 3])
                .toFloat()
                .div(tf.scalar(255))
                .expandDims();
            
            images.push(imageTensor);
        } catch (error) {
            console.error(`Failed to process image ${file}:`, error);
        }
    }
    return images;
}

// Evaluate the model
async function evaluateModel() {
    const model = await tf.loadLayersModel('file://./gender_detection_model/model.json');

    // Compile the model (replace 'loss' and 'optimizer' with your model's configuration)
    model.compile({
        loss: 'binaryCrossentropy',
        optimizer: 'adam',
        metrics: ['accuracy']
    });
    
    const testImages = await loadImageFromDir(path.join(testDataDir, 'male'));

    // Ensure there are images to evaluate
    if (testImages.length === 0) {
        console.error('No valid images found for evaluation.');
        return;
    }

    const testLabels = tf.tensor1d(new Array(testImages.length).fill(0));  // 0 for male

    // Concatenate the images into a single tensor
    const imagesTensor = tf.concat(testImages);
    
    const results = model.evaluate(imagesTensor, testLabels, { batchSize });
    
    // Since evaluate returns a tensor, we need to fetch the data from it
    const accuracy = (await results[1].data())[0];
    console.log(`Test Accuracy: ${accuracy}`);
}

evaluateModel();
