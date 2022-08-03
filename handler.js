const tf = require('@tensorflow/tfjs')
const mobilenet = require('@tensorflow-models/mobilenet');
global.fetch = require('node-fetch');

const image = require('get-image-data');
let response;

module.exports.getname = async (event, context, callback) => {
  try {
    let url = event.url;
    response = whatIsThis(url);
    function whatIsThis(url) {
      return new Promise((resolve, reject) => {
        image(url, async (err, image) => {
          if (err) {
            reject(err);
          } else {
            const channelCount = 3;
            const pixelCount = image.width * image.height;
            const vals = new Int32Array(pixelCount * channelCount);
            let pixels = image.data;
            for (let i = 0; i < pixelCount; i++) {
              for (let k = 0; k < channelCount; k++) {
                vals[i * channelCount + k] = pixels[i * 4 + k];
              }
            }
            const outputShape = [image.height, image.width, channelCount];
            const input = tf.tensor3d(vals, outputShape, "int32");
            const model = await mobilenet.load();
            let temp = await model.classify(input);
            resolve(temp);
          }
        });
      });
    }
  } catch (err) {
    console.log(err);
  }
  return response;
  };

