import { Connection, Client } from "@temporalio/client";
import { nanoid } from "nanoid";

async function run() {
  // Connect to the default Server location
  // const connection = await Connection.connect({
  //   address: process.env?.TEMPORAL_ADDRESS,
  // });

  const connection = await Connection.connect({
    address: "54.251.96.179:7233", // ✅ Replace with your Temporal server's IP
  });
  // In production, pass options to configure TLS and other settings:
  // {
  //   address: 'foo.bar.tmprl.cloud',
  //   tls: {}
  // }

  const client = new Client({
    connection,
    // namespace: 'foo.bar', // connects to 'default' namespace if not specified
  });
  console.log("client: ", client);

  const uuid = nanoid();

  // /*const payload = {
  //   taskType: "text-classification",
  //   modelFramework: "pytorch",
  //   modelArchitecture: "distilbert",
  //   modelWeightUrl:
  //     "http://localhost:5500/temporal/scripts/text-classification/distilbert/pkl/weights/model.pkl",
  //   // modelWeightUrl:
  //   //   "http://host.docker.internal:5500/temporal/scripts/text-classification/distilbert/pkl/weights/model.pkl",
  //   uuid: uuid,
  // };*/

  // const payload = {
  //   dataType: "structured",
  //   taskType: "tabular-regression",
  //   modelFramework: "scikit-learn",
  //   modelArchitecture: "linear-regression",
  //   targetColumn: "Weight",
  //   experimentName: "fishweight-linear-regression",
  //   modelWeightUrl:
  //     "http://localhost:5500/temporal/scripts/structured/tabular-regression/scikit-learn/linear-regression/weights/fish.pkl",
  //   modelDatasetUrl:
  //     "http://localhost:5500/temporal/scripts/structured/tabular-regression/scikit-learn/linear-regression/datasets/fish.csv",
  //   uuid: uuid,
  // };


  // const payload = {
  //   dataType: "structured",
  //   taskType: "tabular-regression",
  //   modelFramework: "scikit-learn",
  //   modelArchitecture: "linear-regression",
  //   targetColumn: "Weight",
  //   experimentName: "heart-linear-regression",
  //   modelWeightUrl:
  //     "http://127.0.0.1:5501/temporal/scripts/structured/tabular-regression/scikit-learn/linear-regression/weights/heart_model.pkl",
  //   modelDatasetUrl:
  //     "http://127.0.0.1:5501/temporal/scripts/structured/tabular-regression/scikit-learn/linear-regression/datasets/heart.csv",
  //   uuid: uuid,
  // };

  /*const payload = {
    dataType: "unstructured",
    taskType: "image-classification",
    modelFramework: "onnx",
    modelArchitecture: "resnet",
    experimentName: "ResNet-Image-Classification-onnx",
    modelWeightUrl:
      "http://localhost:5500/temporal/scripts/unstructured/image-classification/onnx/resnet/weights/resnet50-v1-7.onnx",
    modelDatasetUrl:
      "http://localhost:5500/temporal/scripts/unstructured/image-classification/onnx/resnet/datasets/data.csv",
    imageZipUrl:
      "http://localhost:5500/temporal/scripts/unstructured/image-classification/onnx/resnet/datasets/images.zip",
    dataLabelUrl:
      "http://localhost:5500/temporal/scripts/unstructured/image-classification/onnx/resnet/datasets/imagenet_classes.txt",
    uuid: uuid,
  };*/

  //  const payload = {
  //   dataType: "unstructured",
  //   taskType: "image-classification",
  //   modelFramework: "tensorflow",
  //   modelArchitecture: "convolutional-neural-network",
  //   experimentName: "tensorflow-Image-Classification",
  //   modelWeightUrl:{

  //     path: "http://localhost:5500/temporal/scripts/unstructured/image-classification/tensorflow/convolutional-neural-network/weights/vgg16_model.h5",
  //   },
  //   modelDatasetUrl:[{
  //     Key:"path",
  //     Value: "http://localhost:5500/temporal/scripts/unstructured/image-classification/tensorflow/convolutional-neural-network/datasets/test.zip",
  //   }],
  //   uuid: uuid,
  // };

  // const payload = {
  //   dataType: "structured",
  //   taskType: "tabular-classification",
  //   modelFramework: "xgboost",
  //   modelArchitecture: "tree-based",
  //   targetColumn: "IncidentGrade",
  //   experimentName: "Microsoft-Security-Incident-Prediction",
  //   modelWeightUrl:
  //     "http://127.0.0.1:5501/temporal/scripts/structured/tabular-classification/xgboost/tree-based/weights/weight.pkl",
  //   modelDatasetUrl:
  //     "http://127.0.0.1:5501/temporal/scripts/structured/tabular--classification/xgboost/tree-based/datasets/dataset.csv",
  //   uuid: uuid,
  // };

 
  const handle = await client.workflow.start("runEval", {
    taskQueue: "evaluation",
    // type inference works! args: [name: string]
    args: [payload],
    // in practice, use a meaningful business ID, like customerId or transactionId
    workflowId: "workflow-" + uuid,
  });
  console.log(`Started workflow ${handle.workflowId}`);

  console.log('handle', handle);

  // optional: wait for client result
  console.log(await handle.result()); // Hello, Temporal!
}



run().catch((err) => {
  console.error(err);
  process.exit(1);
});
