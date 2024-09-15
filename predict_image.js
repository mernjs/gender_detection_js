const tf = require('@tensorflow/tfjs-node');
const { createCanvas, loadImage } = require('canvas');
const argparse = require('argparse');

// Command line argument parsing
const parser = new argparse.ArgumentParser({ description: 'Predict gender from an image' });
parser.addArgument('--image', { required: true, help: 'Path to the image file' });
const args = parser.parseArgs();

// Load and preprocess the image
async function preprocessImage(imagePath) {
    const img = await loadImage(imagePath);
    const canvas = createCanvas(150, 150);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 150, 150);

    const imageBuffer = canvas.toBuffer('image/png'); // Get image buffer
    const imageTensor = tf.node.decodeImage(imageBuffer, 3); // Decode image buffer into tensor
    const resizedImageTensor = tf.image.resizeBilinear(imageTensor, [150, 150]).toFloat().div(tf.scalar(255)).expandDims(0);
    return resizedImageTensor;
}

// Predict gender
async function predictGender(imagePath) {
    const model = await tf.loadLayersModel('file://./gender_detection_model/model.json');
    
    const imageTensor = await preprocessImage(imagePath);
    const prediction = await model.predict(imageTensor).data();

    const gender = prediction[0] > 0.5 ? 'Male' : 'Female';
    console.log(`The predicted gender is: ${gender}`);
}

predictGender(args.image);
