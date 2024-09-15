const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

// Parameters
const imageSize = [150, 150];
const batchSize = 32;
const epochs = 10;
const dataDir = 'dataset/training';

// Load images from the directory
async function loadImageFromDir(dir) {
    const files = fs.readdirSync(dir);
    const images = [];

    for (let file of files) {
        const filePath = path.join(dir, file);
        try {
            // Check if the file is a valid image type
            if (path.extname(file).toLowerCase() === '.jpg' || path.extname(file).toLowerCase() === '.jpeg') {
                const img = await loadImage(filePath);
                const canvas = createCanvas(imageSize[0], imageSize[1]);
                const ctx = canvas.getContext('2d');

                // Draw image onto canvas and ensure correct dimensions
                ctx.drawImage(img, 0, 0, imageSize[0], imageSize[1]);

                // Extract image data and convert to tensor
                const imageData = ctx.getImageData(0, 0, imageSize[0], imageSize[1]);
                const imageTensor = tf.tensor3d(new Uint8Array(imageData.data), [imageSize[0], imageSize[1], 4])
                    .slice([0, 0, 0], [imageSize[0], imageSize[1], 3])
                    .toFloat()
                    .div(tf.scalar(255))
                    .expandDims();
                images.push(imageTensor);
            } else {
                console.warn(`Unsupported file format for ${file}`);
            }
        } catch (error) {
            console.error(`Error processing image ${file}: ${error.message}`);
        }
    }
    return images;
}

// Build the CNN Model
const model = tf.sequential();
model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', inputShape: [150, 150, 3] }));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu' }));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
model.add(tf.layers.dropout({ rate: 0.5 }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compile the Model
model.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
});

// Train the model
async function trainModel() {
    const trainImages = await loadImageFromDir(path.join(dataDir, 'male'));
    const valImages = await loadImageFromDir(path.join(dataDir, 'female'));

    if (trainImages.length === 0 || valImages.length === 0) {
        throw new Error('No valid images found in the directories');
    }

    const xTrain = tf.concat(trainImages);
    const yTrain = tf.tensor1d(new Array(trainImages.length).fill(0));  // 0 for male

    const xVal = tf.concat(valImages);
    const yVal = tf.tensor1d(new Array(valImages.length).fill(1));  // 1 for female

    await model.fit(xTrain, yTrain, {
        epochs: epochs,
        batchSize: batchSize,
        validationData: [xVal, yVal]
    });

    // Save the model
    await model.save('file://./gender_detection_model');
}

trainModel().then(() => console.log('Training Complete')).catch(console.error);
