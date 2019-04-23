
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

async function prepareDataset() {
  const parseImage = async ([path, label]) => {
    const img = await loadImage(`http://127.0.0.1:8080${path}`)
    
    const image = tf.browser.fromPixels(img)
    // console.log('p: ', path)
    // console.log('img: ', image)
    // await image.print()

    return image
  }

  const imgs = imagesAndLables.map(parseImage);
  // const imgs = await Promise.all(imagesAndLables.map(parseImage));
  const labels = imagesAndLables.map(([path, label]) => label);

  const xDataset = tf.data.array(imgs);
  const yDataset = tf.data.array(labels);
  // yDataset.print()
  const xyDataset = tf.data.zip({xs: xDataset, ys: yDataset})

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
  // console.log('event: ', event)
  const xyDataset = prepareDataset()

  const model = getModel()
  // model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
  const history = await model.fitDataset(xyDataset, {
    epochs: 4,
    callbacks: {onEpochEnd: (epoch, logs) => console.log(logs.loss)}
  });
});

function getModel() {
  const model = tf.sequential();
  
  const IMAGE_WIDTH = 500;
  const IMAGE_HEIGHT = 375;
  const IMAGE_CHANNELS = 3;
  
  // In the first layer of out convolutional neural network we have 
  // to specify the input shape. Then we specify some paramaters for 
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.  
  model.add(tf.layers.maxPooling2d({poolSize: [1, 1], strides: [1, 1]}));
  
  // Repeat another conv2d + maxPooling stack. 
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [1, 1], strides: [1, 1]}));
  
  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 2;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  
  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}
