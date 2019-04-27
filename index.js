
const title = {
  0: 'тарілка',
  1: 'книга'
}
const imagesAndLables = [
  [ '/images/02_5.jpg', 0],
  [ '/images/01_5.jpg', 0],
  [ '/images/03_5.jpg', 0],
  [ '/images/04_5.jpg', 0],
  [ '/images/05_5.jpg', 0],
  [ '/images/06_5.jpg', 0],
  // [ '/images/07_5.jpg', 0],
  // [ '/images/08_5.jpg', 0],
  // [ '/images/09_5.jpg', 0],
  [ '/images/10_5.jpg', 1],
  [ '/images/11_5.jpg', 1],
  [ '/images/12_5.jpg', 1],
  // [ '/images/13_5.jpg', 1],
  // [ '/images/14_5.jpg', 1],
  // [ '/images/15_5.jpg', 1],
  // [ '/images/16_5.jpg', 1],
  // [ '/images/17_5.jpg', 1],
  // [ '/images/18_5.jpg', 1],
  // [ '/images/19_5.jpg', 1],
]

let xDataset, yDataset;
let model

const parseImage = async ([path, label]) => {
  const img = await loadImage(`http://127.0.0.1:8080${path}`)
  
  const image = tf.browser.fromPixels(img)
  return image
}

async function prepareDataset() {
  const imgs = await Promise.all(imagesAndLables.map(parseImage));
  const labels = imagesAndLables.map(([path, label]) => tf.scalar(label));

  xDataset = tf.data.array(imgs);
  yDataset = tf.data.array(labels);
  const xyDataset = tf.data.zip({xs: xDataset, ys: yDataset})
    .batch(8);

  return xyDataset
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      console.log('img: ', img, img.height, img.width)
      resolve(img)
    }
    img.onerror = (err) => {
      console.log('err121: ', err)
      reject(err)
    }
    img.src = src
  })
}

document.querySelector('#btn-load').addEventListener('click', async (event) => {
  const xyDataset = await prepareDataset()

  model = getModel()
  model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
  tfvis.show.modelSummary({name: 'Model Architecture'}, model);
  
  await train(model, xyDataset);
});

document.querySelector('#pred-load').addEventListener('click', async (event) => {
  const img =  await loadImage('http://127.0.0.1:8080/images/02_5.jpg')
  let ten = tf.browser.fromPixels(img)
  ten = ten.reshape([1, 500, 375, 3])

  console.log('Result: ', await model.predictOnBatch(ten).data())
})

function getModel() {
  const model = tf.sequential();
  
  const IMAGE_WIDTH = 500;
  const IMAGE_HEIGHT = 375;
  const IMAGE_CHANNELS = 3;

  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [1, 1], strides: [1, 1]}));

  model.add(tf.layers.flatten());

  const NUM_OUTPUT_CLASSES = 2;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function train(model, xyDataset) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
  const BATCH_SIZE = 1;

  return model.fitDataset(xyDataset, {
    batchSize: BATCH_SIZE,
    epochs: 8,
    shuffle: true,
    callbacks: fitCallbacks
  });
}
