let net;

const imgEl = document.getElementById("img");
const descEl = document.getElementById("descripcion_imagen");
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();


async function app() {
  net = await mobilenet.load();

  let result = await net.classify(imgEl);
  console.log(result);

  displayImagePrediction();


  webcam = await tf.data.webcam(webcamElement);

  while(true){
    const img = await webcam.capture();

    const result = await net.classify(img);

    const activation = net.infer(img, "conv_preds");

    var result2;

    try{
      result2 = await classifier.predictClass(activation);
      const classes = ["Celular","Yo", "Ok", "Rock", "Peace and love"];
      document.getElementById('console2').innerHTML = classes[result2.label-1];
    }
    catch(error){
      console.log(error);
    }

    document.getElementById('console').innerHTML = 'prediction: ' + result[0].className + ' probability: ' + result[0].probability;

    img.dispose();

    await tf.nextFrame();

  }
}

imgEl.onload = async function () {
  displayImagePrediction();
};

async function addExample(classId){
  console.log('Example added');
  const img = await webcam.capture();
  const activation = net.infer(img, true);
  classifier.addExample(activation, classId);

  img.dispose();
}

async function displayImagePrediction() {
  try {
    result = await net.classify(imgEl);
    descEl.innerHTML = JSON.stringify(result);
  } catch (error) {}
}

var count = 0;
async function cambiarImagen() {
  count = count + 1;
  imgEl.src = "https://picsum.photos/200/300?random=" + count;
}

app();
