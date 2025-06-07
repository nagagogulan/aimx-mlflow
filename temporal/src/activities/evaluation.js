import { exec } from "child_process";
import { nanoid } from "nanoid";
import * as k8s from "@kubernetes/client-node";
import path from "path";
// import path from 'path';
import { fileURLToPath } from 'url';
import { Kafka } from "kafkajs";
import * as allfunction from "../kafka/worker.js" ;

const projectRoot = "/app"; // ✅ Container-based 
console.log('PROJECT ROOT:', projectRoot);

export async function helloWorld(options) {
  return "hello world";
}

export async function copyInferenceScriptsOld(options) {
  const tempId = nanoid();
  // const INFERENCE_BASE_DIR = `${process?.env?.EVAL_BASE_DIR}/scripts/text-classification/distilbert/pkl`;
  // console.log("INFERENCE_BASE_DIR: ", path.resolve(__dirname, '../scripts/text-classification/distilbert/pkl'));
const INFERENCE_BASE_DIR = path.join(projectRoot, 'scripts', 'text-classification', 'distilbert', 'pkl');
 
console.log('PROJECT ROOT:', projectRoot);
// console.log('INFERENCE_BASE_DIR:', INFERENCE_BASE_DIR);
  console.log("INFERENCE_BASE_DIR: ", INFERENCE_BASE_DIR);
  const INFERENCE_SCRIPT_PATH = `${INFERENCE_BASE_DIR}/src`;
  const REQUIREMENTS_FILE = `${INFERENCE_BASE_DIR}/requirements.txt`;
  const DOCKER_FILE_DIR = `${INFERENCE_BASE_DIR}/Dockerfile`;

  // const TARGET_DIR = `${process?.env?.EVAL_BASE_DIR}/temporal-runs/${tempId}`;
  const TARGET_DIR = path.join(projectRoot, 'temporal-runs',tempId);
  const MODEL_WEIGHT_DIR = `${TARGET_DIR}/weights`;
  const MODEL_WEIGHT_URL = options?.modelWeightUrl;

  // Step 1: Create the target directory
  console.log(`Creating target directory: ${TARGET_DIR}`);
  await runCommand(`mkdir -p ${TARGET_DIR}`);

  // Step 2: Copy the inference scripts
  console.log(
    `Copying inference scripts from ${INFERENCE_SCRIPT_PATH} to ${TARGET_DIR}`
  );
  await runCommand(`cp -r ${INFERENCE_SCRIPT_PATH} ${TARGET_DIR}`);

  // Step 3: Copy the Dockerfile
  console.log(`Copying Dockerfile from ${DOCKER_FILE_DIR} to ${TARGET_DIR}`);
  await runCommand(`cp ${DOCKER_FILE_DIR} ${TARGET_DIR}`);

  // Step 4: Create the weights directory
  console.log(`Creating weights directory: ${MODEL_WEIGHT_DIR}`);
  await runCommand(`mkdir -p ${MODEL_WEIGHT_DIR}`);

  // Step 5: Copy the model weights
  console.log(
    `Downloading model weights from ${MODEL_WEIGHT_URL} to ${MODEL_WEIGHT_DIR}`
  );
  await runCommand(`curl -o ${MODEL_WEIGHT_DIR}/model.pkl ${MODEL_WEIGHT_URL}`);

  // Step 6: Copy the requirements file
  console.log(`Copying requirements file to ${TARGET_DIR}`);
  await runCommand(`cp ${REQUIREMENTS_FILE} ${TARGET_DIR}`);

  return {
    tempId: tempId,
    targetDir: TARGET_DIR,
  };
}

export async function copyInferenceScripts(options) {
  console.log("📥 [Activity] Received:", JSON.stringify(options, null, 2));
  const tempId = nanoid();
  const INFERENCE_BASE_DIR = `${projectRoot}/scripts/${options?.dataType}/${options?.taskType}/${options?.modelFramework}/${options?.modelArchitecture}`;
  const INFERENCE_SCRIPT_PATH = `${INFERENCE_BASE_DIR}/src`;
  const REQUIREMENTS_FILE = `${INFERENCE_BASE_DIR}/requirements.txt`;
  const DOCKER_FILE_DIR = `${INFERENCE_BASE_DIR}/Dockerfile`;

  const TARGET_DIR = `${projectRoot}/temporal-runs/${tempId}`;
  console.log("TARGET_DIR: ", TARGET_DIR);
  const MODEL_WEIGHT_DIR = `${TARGET_DIR}/weights`;
  console.log("MODEL_WEIGHT_DIR: ", MODEL_WEIGHT_DIR);
  const MODEL_WEIGHT_URL = options?.modelWeightUrl;
  console.log("MODEL_WEIGHT_URL: ", MODEL_WEIGHT_URL);
  const DATASETS_DIR = `${TARGET_DIR}/datasets`;
  console.log("DATASETS_DIR: ", DATASETS_DIR);
  const DATASET_URL = options?.modelDatasetUrl[0];
  console.log("DATASET_URL: ", DATASET_URL);
  // console.log("MODEL_WEIGHT_URL: ", MODEL_WEIGHT_URL);
  if (!MODEL_WEIGHT_URL?.path || !DATASET_URL?.Value) {
    throw new Error("Missing modelWeightUrl.path or modelDatasetUrl[0].Value in input options.");
  }

  const modelFileName = MODEL_WEIGHT_URL.path.split("/").pop();
  const datasetFileName = DATASET_URL.Value.split("/").pop();
  console.log("modelFileName: ", modelFileName);
  console.log("datasetFileName: ", datasetFileName);
  const modelWeightFullPath = path.resolve(MODEL_WEIGHT_URL.path);
  const datasetFullPath = path.resolve(DATASET_URL.Value);

  console.log("modelWeightFullPath: ", modelWeightFullPath);
  console.log("datasetFullPath: ", datasetFullPath);


  let IMAGE_ZIP_URL, DATA_LABEL_URL;
  let imageZipFileName, dataLabelFileName;
  if (
    options?.dataType === "unstructured" &&
    options?.taskType === "image-classification" &&
    options?.modelFramework === "onnx" &&
    options?.modelArchitecture === "resnet"
  ) {
    IMAGE_ZIP_URL = options?.imageZipUrl;
    DATA_LABEL_URL = options?.dataLabelUrl;
    imageZipFileName = new URL(IMAGE_ZIP_URL).pathname.split("/").pop();
    dataLabelFileName = new URL(DATA_LABEL_URL).pathname.split("/").pop();
  }

  // Create the target directory
  console.log(`Creating target directory: ${TARGET_DIR}`);
  await runCommand(`mkdir -p ${TARGET_DIR}`);

  // Copy the inference scripts
  console.log(
    `Copying inference scripts from ${INFERENCE_SCRIPT_PATH} to ${TARGET_DIR}`
  );
  await runCommand(`cp -r ${INFERENCE_SCRIPT_PATH} ${TARGET_DIR}`);

  // Copy the Dockerfile
  console.log(`Copying Dockerfile from ${DOCKER_FILE_DIR} to ${TARGET_DIR}`);
  await runCommand(`cp ${DOCKER_FILE_DIR} ${TARGET_DIR}`);

  // Create the weights directory
  console.log(`Creating weights directory: ${MODEL_WEIGHT_DIR}`);
  await runCommand(`mkdir -p ${MODEL_WEIGHT_DIR}`);

  // Copy the model weights
  console.log(`modelWeightFullPath: ${modelWeightFullPath} ${MODEL_WEIGHT_DIR}/${modelFileName}`);
  if (!modelWeightFullPath || !modelFileName) {
    throw new Error("Invalid model weight path or filename");
  }
  // `curl -o ${MODEL_WEIGHT_DIR}/${modelFileName} ${MODEL_WEIGHT_URL}`
  await runCommand(
    `cp ${modelWeightFullPath} ${MODEL_WEIGHT_DIR}/${modelFileName}`
  );

  // Create the datasets directory
  console.log(`Creating datasets directory: ${DATASETS_DIR}`);
  await runCommand(`mkdir -p ${DATASETS_DIR}`);

  // Copy the datasets
  console.log(`Downloading datasets from ${DATASET_URL} to ${DATASETS_DIR}`);
  console.log(`datasetFullPath: ${datasetFullPath} ${DATASETS_DIR}/${datasetFileName}`);
  // await runCommand(`curl -o ${DATASETS_DIR}/${datasetFileName} ${DATASET_URL}`);\
  await runCommand(
    `cp ${datasetFullPath} ${DATASETS_DIR}/${datasetFileName}`
  );

  // Copy the requirements file
  console.log(`Copying requirements file to ${TARGET_DIR}`);
  await runCommand(`cp ${REQUIREMENTS_FILE} ${TARGET_DIR}`);

  if (
    options?.dataType === "unstructured" &&
    options?.taskType === "image-classification" &&
    options?.modelFramework === "onnx" &&
    options?.modelArchitecture === "resnet"
  ) {
    // Copy the image zip file
    console.log(
      `Downloading image zip file from ${IMAGE_ZIP_URL} to ${DATASETS_DIR}`
    );
    await runCommand(
      // `curl -o ${DATASETS_DIR}/${imageZipFileName} ${IMAGE_ZIP_URL}`
      `cp ${DATASETS_DIR}/${imageZipFileName} ${IMAGE_ZIP_URL}`

    );

    // Copy the data label file
    console.log(
      `Downloading data label file from ${DATA_LABEL_URL} to ${DATASETS_DIR}`
    );
    await runCommand(
      // `curl -o ${DATASETS_DIR}/${dataLabelFileName} ${DATA_LABEL_URL}`
      `cp ${DATASETS_DIR}/${dataLabelFileName} ${DATA_LABEL_URL}`

    );
  }

  return {
    tempId: tempId,
    targetDir: TARGET_DIR,
    weightsPath: `./weights/${modelFileName}`,
    datasetPath: `./datasets/${datasetFileName}`,
    imageZipPath: `./datasets/${imageZipFileName}` || null,
    dataLabelPath: `./datasets/${dataLabelFileName}` || null,
  };
}

export async function buildDockerImage(options) {
  console.log(`Building evaluation container from ${options.targetDir}`);

  // Step 1: Stop and remove existing container if running
  await runCommand("docker rm -f aimx-evaluation || true", options.targetDir);

  // Step 2: Build the Docker image
  await runCommand("docker build -t aimx-evaluation .", options.targetDir);

  return "Docker image built successfully!";
}

export async function runEvaluations(options) {
  const device = "0";
  const modelWeightsPath = process?.env?.WEIGHTS_PATH;
  const mlFlowUrl = process?.env?.MLFLOW_URL;

  console.log(`running evaluation container from ${options.targetDir}`);

  // Run the container with dynamic environment variables
  await runCommand(
    `docker run -d -p 8888:8888 --gpus all --name aimx-evaluation \
              -e DEVICE=${device} \
              -e MODEL_WIGHTS_PATH=${modelWeightsPath} \
              -e MLFLOW_TRACKING_URI=${mlFlowUrl} \
              aimx-evaluation`,
    options.targetDir
  );
  return "Docker container started successfully!";
}

export async function runEvaluationsInCluster(options, inferenceData) {
  const kc = new k8s.KubeConfig();
  kc.loadFromDefault(); // This will load from ~/.kube/config
  const k8sApi = kc.makeApiClient(k8s.CoreV1Api);
  const k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);

  // generate random string that returns 4 characters (only alphabets)
  const randomString = generateRandomString(4).toLocaleLowerCase();

  const namespace = process?.env?.NAMESPACE || "default";
  const jobName = `aimx-evaluation-${randomString}`; // Unique job name

  let containerData;

  if (
    options?.dataType === "structured" &&
    options?.taskType === "tabular-regression" &&
    options?.modelFramework === "scikit-learn" &&
    options?.modelArchitecture === "linear-regression"
  ) {
    containerData = [
      {
        name: "aimx-evaluation",
        image: "aimx-evaluation:latest",
        imagePullPolicy: "Never", // Use local image
        env: [
          {
            name: "MODEL_WIGHTS_PATH",
            value: inferenceData.weightsPath,
          },
          {
            name: "MLFLOW_TRACKING_URI",
            value: process.env.MLFLOW_URL,
          },
          {
            name: "DATASET_PATH",
            value: inferenceData.datasetPath,
          },
          {
            name: "TARGET_COLUMN",
            value: options?.targetColumn || "target",
          },
          {
            name: "EXPERIMENT_NAME",
            value: options?.experimentName || "default_experiment",
          },
        ],
      },
    ];
  } else if (
    options?.dataType === "unstructured" &&
    options?.taskType === "image-classification" &&
    options?.modelFramework === "onnx" &&
    options?.modelArchitecture === "resnet"
  ) {
    containerData = [
      {
        name: "aimx-evaluation",
        image: "aimx-evaluation:latest",
        imagePullPolicy: "Never", // Use local image
        env: [
          {
            name: "MODEL_WIGHTS_PATH",
            value: inferenceData.weightsPath,
          },
          {
            name: "MLFLOW_TRACKING_URI",
            value: process.env.MLFLOW_URL,
          },
          {
            name: "DATASET_PATH",
            value: inferenceData.datasetPath,
          },
          {
            name: "IMAGES_ZIP_PATH",
            value: inferenceData.imageZipPath,
          },
          {
            name: "DATA_LABEL_PATH",
            value: inferenceData.dataLabelPath,
          },
          {
            name: "EXPERIMENT_NAME",
            value: options?.experimentName || "default_experiment",
          },
        ],
      },
    ];
  } else if (
    options?.dataType === "structured" &&
    options?.taskType === "tabular-classification" &&
    options?.modelFramework === "xgboost" &&
    options?.modelArchitecture === "tree-based"
  ) {
    containerData = [
      {
        name: "aimx-evaluation",
        image: "aimx-evaluation:latest",
        imagePullPolicy: "Never", // Use local image
        env: [
          {
            name: "MODEL_WIGHTS_PATH",
            value: inferenceData.weightsPath,
          },
          {
            name: "MLFLOW_TRACKING_URI",
            value: process.env.MLFLOW_URL,
          },
          {
            name: "DATASET_PATH",
            value: inferenceData.datasetPath,
          },
          {
            name: "TARGET_COLUMN",
            value: options?.targetColumn || "target",
          },
          {
            name: "EXPERIMENT_NAME",
            value: options?.experimentName || "default_experiment",
          },
        ],
      },
    ];
  }

  const jobManifest = {
    apiVersion: "batch/v1",
    kind: "Job",
    metadata: {
      name: jobName,
    },
    spec: {
      template: {
        metadata: {
          name: jobName,
        },
        spec: {
          containers: containerData,
          restartPolicy: "Never", // Very important for Jobs
        },
      },
      backoffLimit: 0, // No retries if it fails
    },
  };

  await k8sBatchApi.createNamespacedJob({
    namespace,
    body: jobManifest,
  });

  return {
    jobName: jobName,
    namespace: namespace,
  };
}

export async function waitForJobCompletion(
  jobName,
  namespace,
  timeoutMs = 600000,
  pollInterval = 5000
) {
  const kc = new k8s.KubeConfig();
  kc.loadFromDefault(); // This will load from ~/.kube/config
  const k8sApi = kc.makeApiClient(k8s.CoreV1Api);
  const k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);
  const start = Date.now();

  while (true) {
    const job = await k8sBatchApi.readNamespacedJob({
      name: jobName,
      namespace,
    });

    if (job.status.succeeded === 1) {
      console.log(`✅ Job ${jobName} completed successfully.`);
      return true;
    }

    if (job.status.failed && job.status.failed > 0) {
      throw new Error(
        `❌ Job ${jobName} failed with ${job.status.failed} failures.`
      );
    }

    if (Date.now() - start > timeoutMs) {
      throw new Error(`⏰ Timeout waiting for Job ${jobName} to complete.`);
    }

    await new Promise((resolve) => setTimeout(resolve, pollInterval));
  }
}

const runCommand = (cmd, cwd = process?.env?.DOCKER_FILE_DIR) => {
  return new Promise((resolve, reject) => {
    console.log(`🔹 Executing: ${cmd} (in ${cwd})`); // Log command & directory
    exec(cmd, { cwd: cwd }, (error, stdout, stderr) => {
      if (error) {
        console.error(`❌ Command failed: ${cmd}\nError: ${error.message}`);
        console.error(`Stderr:\n${stderr}`);
        reject(new Error(stderr || error.message));
        return;
      }
      console.log(`✅ Command succeeded:\n${stdout}`);
      resolve(stdout.trim()); // Trim output for cleaner logs
    });
  });
};

const generateRandomString = (length = 4) => {
  return Array.from({ length: length }, () =>
    String.fromCharCode(97 + Math.floor(Math.random() * 26))
  ).join("");
};

export async function sendDocketStatus(uuid, status) {
  const broker = "54.251.96.179:9092";
  const topic = "docket-status";
 
  // Create topic if it doesn't exist
  allfunction.createTopicIfNotExists(broker, topic);
 
  const kafka = new Kafka({ brokers: [broker] });
  const producer = kafka.producer();
  await producer.connect();
 
  const message = {
    uuid,
    status
  };
 
  await producer.send({
    topic,
    messages: [{
      key: uuid,
      value: JSON.stringify(message)
    }]
  });
 
  console.log("Message sent to topic 'docket-status':", message);
 
  await producer.disconnect();
  
}