import tf from "@tensorflow/tfjs";
import mobilenet from "@tensorflow-models/mobilenet";
import fetch from "node-fetch";
globalThis.fetch = fetch;
import image from "get-image-data";

export const lambdaHandler = async (event, context, callback) => {
    var imgurl = JSON.parse(event.body);
    let url = imgurl.url;
    const classification = await classify(url);
    const response = {
      statusCode: 200,
      body: JSON.stringify({
        classification
    })
    };
    callback(null, response);
};

function classify(url) {
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