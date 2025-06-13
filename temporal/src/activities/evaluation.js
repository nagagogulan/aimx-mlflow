import { exec } from "child_process";
import { nanoid } from "nanoid";
import * as k8s from "@kubernetes/client-node";
import path from "path";
// import path from 'path';
import { fileURLToPath } from 'url';
import { Kafka } from "kafkajs";
import * as allfunction from "../kafka/worker.js" ;
import fs from 'fs';
import https from 'https';
import yaml from 'js-yaml';
import Docker from "dockerode";

const projectRoot = "/app"; // ✅ Container-based 
console.log('PROJECT ROOT:', projectRoot);

export async function helloWorld(options) {
  return "hello world";
}

// export async function copyInferenceScriptsOld(options) {
//   const tempId = nanoid();
//   // const INFERENCE_BASE_DIR = `${process?.env?.EVAL_BASE_DIR}/scripts/text-classification/distilbert/pkl`;
//   // console.log("INFERENCE_BASE_DIR: ", path.resolve(__dirname, '../scripts/text-classification/distilbert/pkl'));
// const INFERENCE_BASE_DIR = path.join(projectRoot, 'scripts', 'text-classification', 'distilbert', 'pkl');
 
// console.log('PROJECT ROOT:', projectRoot);
// // console.log('INFERENCE_BASE_DIR:', INFERENCE_BASE_DIR);
//   console.log("INFERENCE_BASE_DIR: ", INFERENCE_BASE_DIR);
//   const INFERENCE_SCRIPT_PATH = `${INFERENCE_BASE_DIR}/src`;
//   const REQUIREMENTS_FILE = `${INFERENCE_BASE_DIR}/requirements.txt`;
//   const DOCKER_FILE_DIR = `${INFERENCE_BASE_DIR}/Dockerfile`;

//   // const TARGET_DIR = `${process?.env?.EVAL_BASE_DIR}/temporal-runs/${tempId}`;
//   const TARGET_DIR = path.join(projectRoot, 'temporal-runs',tempId);
//   const MODEL_WEIGHT_DIR = `${TARGET_DIR}/weights`;
//   const MODEL_WEIGHT_URL = options?.modelWeightUrl;

//   // Step 1: Create the target directory
//   console.log(`Creating target directory: ${TARGET_DIR}`);
//   await runCommand(`mkdir -p ${TARGET_DIR}`);

//   // Step 2: Copy the inference scripts
//   console.log(
//     `Copying inference scripts from ${INFERENCE_SCRIPT_PATH} to ${TARGET_DIR}`
//   );
//   await runCommand(`cp -r ${INFERENCE_SCRIPT_PATH} ${TARGET_DIR}`);

//   // Step 3: Copy the Dockerfile
//   console.log(`Copying Dockerfile from ${DOCKER_FILE_DIR} to ${TARGET_DIR}`);
//   await runCommand(`cp ${DOCKER_FILE_DIR} ${TARGET_DIR}`);

//   // Step 4: Create the weights directory
//   console.log(`Creating weights directory: ${MODEL_WEIGHT_DIR}`);
//   await runCommand(`mkdir -p ${MODEL_WEIGHT_DIR}`);

//   // Step 5: Copy the model weights
//   console.log(
//     `Downloading model weights from ${MODEL_WEIGHT_URL} to ${MODEL_WEIGHT_DIR}`
//   );
//   await runCommand(`curl -o ${MODEL_WEIGHT_DIR}/model.pkl ${MODEL_WEIGHT_URL}`);

//   // Step 6: Copy the requirements file
//   console.log(`Copying requirements file to ${TARGET_DIR}`);
//   await runCommand(`cp ${REQUIREMENTS_FILE} ${TARGET_DIR}`);

//   return {
//     tempId: tempId,
//     targetDir: TARGET_DIR,
//   };
// }

// export async function copyInferenceScripts(options) {
//   console.log("📥 [Activity] Received:", JSON.stringify(options, null, 2));
//   const tempId = nanoid();
//   const INFERENCE_BASE_DIR = `${projectRoot}/scripts/${options?.dataType}/${options?.taskType}/${options?.modelFramework}/${options?.modelArchitecture}`;
//   const INFERENCE_SCRIPT_PATH = `${INFERENCE_BASE_DIR}/src`;
//   const REQUIREMENTS_FILE = `${INFERENCE_BASE_DIR}/requirements.txt`;
//   const DOCKER_FILE_DIR = `${INFERENCE_BASE_DIR}/Dockerfile`;

//   const TARGET_DIR = `${projectRoot}/temporal-runs/${tempId}`;
//   console.log("TARGET_DIR: ", TARGET_DIR);
//   const MODEL_WEIGHT_DIR = `${TARGET_DIR}/weights`;
//   console.log("MODEL_WEIGHT_DIR: ", MODEL_WEIGHT_DIR);
//   const MODEL_WEIGHT_URL = options?.modelWeightUrl;
//   console.log("MODEL_WEIGHT_URL: ", MODEL_WEIGHT_URL);
//   const DATASETS_DIR = `${TARGET_DIR}/datasets`;
//   console.log("DATASETS_DIR: ", DATASETS_DIR);
//   const DATASET_URL = options?.modelDatasetUrl[0];
//   console.log("DATASET_URL: ", DATASET_URL);
//   // console.log("MODEL_WEIGHT_URL: ", MODEL_WEIGHT_URL);
//   if (!MODEL_WEIGHT_URL?.path || !DATASET_URL?.Value) {
//     throw new Error("Missing modelWeightUrl.path or modelDatasetUrl[0].Value in input options.");
//   }

//   const modelFileName = MODEL_WEIGHT_URL.path.split("/").pop();
//   const datasetFileName = DATASET_URL.Value.split("/").pop();
//   console.log("modelFileName: ", modelFileName);
//   console.log("datasetFileName: ", datasetFileName);
//   const modelWeightFullPath = path.resolve(MODEL_WEIGHT_URL.path);
//   const datasetFullPath = path.resolve(DATASET_URL.Value);

//   console.log("modelWeightFullPath: ", modelWeightFullPath);
//   console.log("datasetFullPath: ", datasetFullPath);


//   let IMAGE_ZIP_URL, DATA_LABEL_URL;
//   let imageZipFileName, dataLabelFileName;
//   if (
//     options?.dataType === "unstructured" &&
//     options?.taskType === "image-classification" &&
//     options?.modelFramework === "onnx" &&
//     options?.modelArchitecture === "resnet"
//   ) {
//     IMAGE_ZIP_URL = options?.imageZipUrl;
//     DATA_LABEL_URL = options?.dataLabelUrl;
//     imageZipFileName = new URL(IMAGE_ZIP_URL).pathname.split("/").pop();
//     dataLabelFileName = new URL(DATA_LABEL_URL).pathname.split("/").pop();
//   }

//   // Create the target directory
//   console.log(`Creating target directory: ${TARGET_DIR}`);
//   await runCommand(`mkdir -p ${TARGET_DIR}`);

//   // Copy the inference scripts
//   console.log(
//     `Copying inference scripts from ${INFERENCE_SCRIPT_PATH} to ${TARGET_DIR}`
//   );
//   await runCommand(`cp -r ${INFERENCE_SCRIPT_PATH} ${TARGET_DIR}`);

//   // Copy the Dockerfile
//   console.log(`Copying Dockerfile from ${DOCKER_FILE_DIR} to ${TARGET_DIR}`);
//   await runCommand(`cp ${DOCKER_FILE_DIR} ${TARGET_DIR}`);

//   // Create the weights directory
//   console.log(`Creating weights directory: ${MODEL_WEIGHT_DIR}`);
//   await runCommand(`mkdir -p ${MODEL_WEIGHT_DIR}`);

//   // Copy the model weights
//   console.log(`modelWeightFullPath: ${modelWeightFullPath} ${MODEL_WEIGHT_DIR}/${modelFileName}`);
//   if (!modelWeightFullPath || !modelFileName) {
//     throw new Error("Invalid model weight path or filename");
//   }
//   // `curl -o ${MODEL_WEIGHT_DIR}/${modelFileName} ${MODEL_WEIGHT_URL}`
//   await runCommand(
//     `cp ${modelWeightFullPath} ${MODEL_WEIGHT_DIR}/${modelFileName}`
//   );

//   // Create the datasets directory
//   console.log(`Creating datasets directory: ${DATASETS_DIR}`);
//   await runCommand(`mkdir -p ${DATASETS_DIR}`);

//   // Copy the datasets
//   console.log(`Downloading datasets from ${DATASET_URL} to ${DATASETS_DIR}`);
//   console.log(`datasetFullPath: ${datasetFullPath} ${DATASETS_DIR}/${datasetFileName}`);
//   // await runCommand(`curl -o ${DATASETS_DIR}/${datasetFileName} ${DATASET_URL}`);\
//   await runCommand(
//     `cp ${datasetFullPath} ${DATASETS_DIR}/${datasetFileName}`
//   );

//   // Copy the requirements file
//   console.log(`Copying requirements file to ${TARGET_DIR}`);
//   await runCommand(`cp ${REQUIREMENTS_FILE} ${TARGET_DIR}`);

//   if (
//     options?.dataType === "unstructured" &&
//     options?.taskType === "image-classification" &&
//     options?.modelFramework === "onnx" &&
//     options?.modelArchitecture === "resnet"
//   ) {
//     // Copy the image zip file
//     console.log(
//       `Downloading image zip file from ${IMAGE_ZIP_URL} to ${DATASETS_DIR}`
//     );
//     await runCommand(
//       // `curl -o ${DATASETS_DIR}/${imageZipFileName} ${IMAGE_ZIP_URL}`
//       `cp ${DATASETS_DIR}/${imageZipFileName} ${IMAGE_ZIP_URL}`

//     );

//     // Copy the data label file
//     console.log(
//       `Downloading data label file from ${DATA_LABEL_URL} to ${DATASETS_DIR}`
//     );
//     await runCommand(
//       // `curl -o ${DATASETS_DIR}/${dataLabelFileName} ${DATA_LABEL_URL}`
//       `cp ${DATASETS_DIR}/${dataLabelFileName} ${DATA_LABEL_URL}`

//     );
//   }
//   console.log("final outputt::: to be snet ",
//     tempId,
//     TARGET_DIR,
//     `./weights/${modelFileName}`,
//     `./datasets/${datasetFileName}`,
//     `./datasets/${imageZipFileName}` || null,
//     `./datasets/${dataLabelFileName}` || null
//   )

//   return {
//     tempId: tempId,
//     targetDir: TARGET_DIR,
//     weightsPath: `./weights/${modelFileName}`,
//     datasetPath: `./datasets/${datasetFileName}`,
//     imageZipPath: `./datasets/${imageZipFileName}` || null,
//     dataLabelPath: `./datasets/${dataLabelFileName}` || null,
//   };
// }

export async function copyInferenceScripts(options) {
  const tempId = nanoid();
  console.log(`📥 [copyInferenceScripts] Received options for ID: ${tempId}`);
  
  try {
    const {
      dataType,
      taskType,
      modelFramework,
      modelArchitecture,
      modelWeightUrl,
      modelDatasetUrl,
      imageZipUrl,
      dataLabelUrl,
    } = options;

    if (!modelWeightUrl?.path) {
      console.error("❌ Missing 'modelWeightUrl.path' in input options");
      throw new Error("Missing required field: modelWeightUrl.path");
    }

    const datasetEntry = modelDatasetUrl?.[0];
    if (!datasetEntry?.Value) {
      console.error("❌ Missing 'modelDatasetUrl[0].Value' in input options");
      throw new Error("Missing required field: modelDatasetUrl[0].Value");
    }

    const INFERENCE_BASE_DIR = `${projectRoot}/scripts/${dataType}/${taskType}/${modelFramework}/${modelArchitecture}`;
    const INFERENCE_SCRIPT_PATH = `${INFERENCE_BASE_DIR}/src`;
    const REQUIREMENTS_FILE = `${INFERENCE_BASE_DIR}/requirements.txt`;
    const DOCKER_FILE_DIR = `${INFERENCE_BASE_DIR}/Dockerfile`;

    const TARGET_DIR = `${projectRoot}/temporal-runs/${tempId}`;
    const MODEL_WEIGHT_DIR = `${TARGET_DIR}/weights`;
    const DATASETS_DIR = `${TARGET_DIR}/datasets`;

    const modelFileName = path.basename(modelWeightUrl.path);
    const datasetFileName = path.basename(datasetEntry.Value);

    const modelWeightFullPath = path.resolve(modelWeightUrl.path);
    const datasetFullPath = path.resolve(datasetEntry.Value);

    console.log(`[${tempId}] Creating target directory: ${TARGET_DIR}`);
    await runCommand(`mkdir -p ${TARGET_DIR}`);

    console.log(`[${tempId}] Copying inference scripts from ${INFERENCE_SCRIPT_PATH}`);
    await runCommand(`cp -r ${INFERENCE_SCRIPT_PATH} ${TARGET_DIR}`);

    console.log(`[${tempId}] Copying Dockerfile`);
    await runCommand(`cp ${DOCKER_FILE_DIR} ${TARGET_DIR}`);

    console.log(`[${tempId}] Creating weights directory`);
    await runCommand(`mkdir -p ${MODEL_WEIGHT_DIR}`);

    console.log(`[${tempId}] Copying model weight: ${modelFileName}`);
    await runCommand(`cp ${modelWeightFullPath} ${MODEL_WEIGHT_DIR}/${modelFileName}`);

    console.log(`[${tempId}] Creating datasets directory`);
    await runCommand(`mkdir -p ${DATASETS_DIR}`);

    console.log(`[${tempId}] Copying dataset: ${datasetFileName}`);
    await runCommand(`cp ${datasetFullPath} ${DATASETS_DIR}/${datasetFileName}`);

    console.log(`[${tempId}] Copying requirements.txt`);
    await runCommand(`cp ${REQUIREMENTS_FILE} ${TARGET_DIR}`);

    let imageZipPath = null;
    let dataLabelPath = null;

    if (
      dataType === "unstructured" &&
      taskType === "image-classification" &&
      modelFramework === "onnx" &&
      modelArchitecture === "resnet"
    ) {
      const imageZipFileName = path.basename(new URL(imageZipUrl).pathname);
      const dataLabelFileName = path.basename(new URL(dataLabelUrl).pathname);

      console.log(`[${tempId}] Copying image zip: ${imageZipFileName}`);
      await runCommand(`cp ${DATASETS_DIR}/${imageZipFileName} ${imageZipUrl}`);

      console.log(`[${tempId}] Copying label file: ${dataLabelFileName}`);
      await runCommand(`cp ${DATASETS_DIR}/${dataLabelFileName} ${dataLabelUrl}`);

      imageZipPath = `./datasets/${imageZipFileName}`;
      dataLabelPath = `./datasets/${dataLabelFileName}`;
    }

    console.log(`✅ [${tempId}] All files copied successfully.`);

    return {
      tempId,
      targetDir: TARGET_DIR,
      weightsPath: `./weights/${modelFileName}`,
      datasetPath: `./datasets/${datasetFileName}`,
      imageZipPath,
      dataLabelPath,
    };
  } catch (err) {
    console.error(`❌ [${tempId}] Failed in copyInferenceScripts:`, err.message);
    throw err;
  }
}

// export async function buildDockerImage(options) {
//   console.log(`Building evaluation container from ${options.targetDir}`);

//   // Step 1: Stop and remove existing container if running
//   await runCommand("docker rm -f aimx-evaluation || true", options.targetDir);

//   // Step 2: Build the Docker image
//   await runCommand("docker build -t aimx-evaluation .", options.targetDir);

//   await runCommand("docker tag aimx-evaluation:latest nagagogulan/aimx-evaluation:latest", options.targetDir);
//   await runCommand("docker push nagagogulan/aimx-evaluation:latest", options.targetDir);

//   return "Docker image built successfully!";
// }

export async function buildDockerImage(options) {
  const dir = options.targetDir;

  try {
    console.log(`🔨 [buildDockerImage] Starting build in directory: ${dir}`);

   const username = process?.env?.DOCKER_HUB_USERNAME;
    const password = process?.env?.DOCKER_HUB_PASSWORD;

    if (!username || !password) {
      throw new Error("Docker Hub credentials are missing in environment variables.");
    }

    // Step 1: Remove any existing container
    await runCommand("docker rmi -f aimx-evaluation || true", dir);
    await runCommand("docker rmi -f nagagogulan/aimx-evaluation:latest || true", dir);

    // Step 2: Build the Docker image
    await runCommand("docker build -t aimx-evaluation .", dir);
    console.log(`✅ Docker image built in ${dir}`);

     // Step 3: Log current containers


    // Step 3: Tag and push the image
    await runCommand("docker tag aimx-evaluation:latest nagagogulan/aimx-evaluation:latest", dir);

    await runCommand(`echo ${password} | docker login -u ${username} --password-stdin`, dir);

    await runCommand("docker push nagagogulan/aimx-evaluation:latest", dir);
    console.log(`✅ Docker image pushed to nagagogulan/aimx-evaluation:latest`);

    return "Docker image built successfully!";
  } catch (error) {
    console.error(`❌ [buildDockerImage] Failed to build Docker image in ${dir}: ${error.message}`);
    throw error;
  }
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

// export async function runEvaluationsInCluster(options, inferenceData) {
//   const kc = new k8s.KubeConfig();
//   // kc.loadFromDefault(); // This will load from ~/.kube/config
//   kc.loadFromFile(loadPatchedMinikubeConfig());
  

//   const k8sApi = kc.makeApiClient(k8s.CoreV1Api);
//   const k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);

//   // generate random string that returns 4 characters (only alphabets)
//   const randomString = generateRandomString(4).toLocaleLowerCase();

//   const namespace = process?.env?.NAMESPACE || "default";
//   const jobName = `aimx-evaluation-${randomString}`; // Unique job name

//   let containerData;

//   if (
//     options?.dataType === "structured" &&
//     options?.taskType === "tabular-regression" &&
//     options?.modelFramework === "scikit-learn" &&
//     options?.modelArchitecture === "linear-regression"
//   ) {
//     containerData = [
//       {
//         name: "aimx-evaluation",
//         image: "nagagogulan/aimx-evaluation:latest",
//         imagePullPolicy: "Never", // Use local image
//         env: [
//           {
//             name: "MODEL_WIGHTS_PATH",
//             value: inferenceData.weightsPath,
//           },
//           {
//             name: "MLFLOW_TRACKING_URI",
//             value: process.env.MLFLOW_URL,
//           },
//           {
//             name: "DATASET_PATH",
//             value: inferenceData.datasetPath,
//           },
//           {
//             name: "TARGET_COLUMN",
//             value: options?.targetColumn || "target",
//           },
//           {
//             name: "EXPERIMENT_NAME",
//             value: options?.experimentName || "default_experiment",
//           },
//         ],
//       },
//     ];
//   } else if (
//     options?.dataType === "unstructured" &&
//     options?.taskType === "image-classification" &&
//     options?.modelFramework === "onnx" &&
//     options?.modelArchitecture === "resnet"
//   ) {
//     containerData = [
//       {
//         name: "aimx-evaluation",
//         image: "nagagogulan/aimx-evaluation:latest",
//         imagePullPolicy: "Never", // Use local image
//         env: [
//           {
//             name: "MODEL_WIGHTS_PATH",
//             value: inferenceData.weightsPath,
//           },
//           {
//             name: "MLFLOW_TRACKING_URI",
//             value: process.env.MLFLOW_URL,
//           },
//           {
//             name: "DATASET_PATH",
//             value: inferenceData.datasetPath,
//           },
//           {
//             name: "IMAGES_ZIP_PATH",
//             value: inferenceData.imageZipPath,
//           },
//           {
//             name: "DATA_LABEL_PATH",
//             value: inferenceData.dataLabelPath,
//           },
//           {
//             name: "EXPERIMENT_NAME",
//             value: options?.experimentName || "default_experiment",
//           },
//         ],
//       },
//     ];
//   } else if (
//     options?.dataType === "structured" &&
//     options?.taskType === "tabular-classification" &&
//     options?.modelFramework === "xgboost" &&
//     options?.modelArchitecture === "tree-based"
//   ) {
//     containerData = [
//       {
//         name: "aimx-evaluation",
//         image: "nagagogulan/aimx-evaluation:latest",
//         imagePullPolicy: "Never", // Use local image
//         env: [
//           {
//             name: "MODEL_WIGHTS_PATH",
//             value: inferenceData.weightsPath,
//           },
//           {
//             name: "MLFLOW_TRACKING_URI",
//             value: process.env.MLFLOW_URL,
//           },
//           {
//             name: "DATASET_PATH",
//             value: inferenceData.datasetPath,
//           },
//           {
//             name: "TARGET_COLUMN",
//             value: options?.targetColumn || "target",
//           },
//           {
//             name: "EXPERIMENT_NAME",
//             value: options?.experimentName || "default_experiment",
//           },
//         ],
//       },
//     ];
//   }

//   const jobManifest = {
//     apiVersion: "batch/v1",
//     kind: "Job",
//     metadata: {
//       name: jobName,
//     },
//     spec: {
//       template: {
//         metadata: {
//           name: jobName,
//         },
//         spec: {
//           containers: containerData,
//           restartPolicy: "Never", // Very important for Jobs
//         },
//       },
//       backoffLimit: 0, // No retries if it fails
//     },
//   };

//   await k8sBatchApi.createNamespacedJob({
//     namespace,
//     body: jobManifest,
//   });

//   return {
//     jobName: jobName,
//     namespace: namespace,
//   };
// }

export async function runEvaluationsInCluster(options, inferenceData) {

  await cleanMinikubeDockerResources();
  const randomString = generateRandomString(4).toLowerCase();
  const namespace = process?.env?.NAMESPACE || "default";
  const jobName = `aimx-evaluation-${randomString}`;

  try {
    console.log(`🧠 [runEvaluationsInCluster] Creating job ${jobName} in namespace '${namespace}'`);

    const kc = new k8s.KubeConfig();
    const kubePath = loadPatchedMinikubeConfig();
    kc.loadFromFile(kubePath);

    const cluster = kc.getCurrentCluster();
    const user = kc.getCurrentUser();

    const httpsAgent = new https.Agent({
      ca: fs.readFileSync(cluster?.caFile),
      cert: fs.readFileSync(user?.certFile),
      key: fs.readFileSync(user?.keyFile),
      rejectUnauthorized: true,
    });

    const k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);
    k8sBatchApi.basePath = cluster?.server;
    k8sBatchApi.requestOptions = { agent: httpsAgent };

    const containerData = getContainerEnvConfig(options, inferenceData);

    const jobManifest = {
      apiVersion: "batch/v1",
      kind: "Job",
      metadata: { name: jobName },
      spec: {
        template: {
          metadata: { name: jobName },
          spec: {
            containers: containerData,
            restartPolicy: "Never"
          }
        },
        backoffLimit: 0
      }
    };

    await k8sBatchApi.createNamespacedJob({ namespace, body: jobManifest });

    console.log(`✅ Kubernetes job '${jobName}' created successfully.`);
    return { jobName, namespace };
  } catch (error) {
    console.error(`❌ Failed to create Kubernetes job '${jobName}':`, error.message);
    throw error;
  }
}

function getContainerEnvConfig(options, inferenceData) {
  console.log(`📦 [getContainerEnvConfig] Generating container environment configuration for options:`, options, inferenceData);
  const baseEnv = [
    { name: "MODEL_WIGHTS_PATH", value: inferenceData.weightsPath },
    { name: "MLFLOW_TRACKING_URI", value: process.env.MLFLOW_URL },
    { name: "DATASET_PATH", value: inferenceData.datasetPath },
    { name: "EXPERIMENT_NAME", value: options.experimentName || "Microsoft-Security-Incident-Prediction" },
    { name: "TARGET_COLUMN", value: "IncidentGrade"}
  ];

  // if (options.targetColumn) {
  //   baseEnv.push({ name: "TARGET_COLUMN", value: "target"});
  // }

  if (
    options.dataType === "unstructured" &&
    options.taskType === "image-classification"
  ) {
    baseEnv.push(
      { name: "IMAGES_ZIP_PATH", value: inferenceData.imageZipPath },
      { name: "DATA_LABEL_PATH", value: inferenceData.dataLabelPath }
    );
  }

  return [{
    name: "aimx-evaluation",
    image: "nagagogulan/aimx-evaluation:latest",
    imagePullPolicy: "IfNotPresent",
    env: baseEnv,
    workingDir: "/app"
  }];
}



// export async function waitForJobCompletion(
//   jobName,
//   namespace,
//   timeoutMs = 600000,
//   pollInterval = 5000
// ) {
//   const kc = new k8s.KubeConfig();
//   // kc.loadFromDefault(); // This will load from ~/.kube/config
//     kc.loadFromFile(loadPatchedMinikubeConfig()); // ✅ Use patched kubeconfig
//   const k8sApi = kc.makeApiClient(k8s.CoreV1Api);
//   const k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);
//   const start = Date.now();

//   while (true) {
//     const job = await k8sBatchApi.readNamespacedJob({
//       name: jobName,
//       namespace,
//     });

//     if (job.status.succeeded === 1) {
//       console.log(`✅ Job ${jobName} completed successfully.`);
//       return true;
//     }

//     if (job.status.failed && job.status.failed > 0) {
//       throw new Error(
//         `❌ Job ${jobName} failed with ${job.status.failed} failures.`
//       );
//     }

//     if (Date.now() - start > timeoutMs) {
//       throw new Error(`⏰ Timeout waiting for Job ${jobName} to complete.`);
//     }

//     await new Promise((resolve) => setTimeout(resolve, pollInterval));
//   }
// }

// export async function waitForJobCompletion(
//   jobName,
//   namespace,
//   timeoutMs = 6000000, // 10 minutes
//   pollInterval = 50000  // 5 seconds
// ) {
//   console.log(`⏳ [waitForJobCompletion] Monitoring job: '${jobName}' in namespace: '${namespace}'`);

//   const kc = new k8s.KubeConfig();
//   const start = Date.now();

//   try {
//     kc.loadFromFile(loadPatchedMinikubeConfig());
//     const k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);

//     while (true) {
//       try {
//         const jobResp = await k8sBatchApi.readNamespacedJob({ name: jobName, namespace });
//         const jobStatus = jobResp.body?.status;

//         console.log(`🔍 [Status] job=${jobName}, succeeded=${jobStatus?.succeeded || 0}, failed=${jobStatus?.failed || 0}`);

//         if (jobStatus?.succeeded === 1) {
//           console.log(`✅ [Success] Job '${jobName}' completed successfully.`);
//           return true;
//         }

//         if (jobStatus?.failed && jobStatus.failed > 0) {
//           console.error(`❌ [Failure] Job '${jobName}' failed with ${jobStatus.failed} attempt(s).`);
//           throw new Error(`Job '${jobName}' failed with ${jobStatus.failed} failures.`);
//         }

//       } catch (apiErr) {
//         const statusCode = apiErr?.response?.statusCode;
//         const rawBody = apiErr?.response?.body;

//         if (statusCode === 404) {
//           console.warn(`⚠️ [NotFound] Job '${jobName}' not found. It may not be created yet or was deleted.`);
//         } else {
//           console.error(`❌ [API Error] Failed to fetch job status. Code=${statusCode}`);
//           if (rawBody) {
//             try {
//               const parsed = JSON.parse(rawBody);
//               console.error(`📄 [ErrorBody]: ${JSON.stringify(parsed, null, 2)}`);
//             } catch {
//               console.error(`📄 [RawBody]: ${rawBody}`);
//             }
//           }
//           throw apiErr;
//         }
//       }

//       if (Date.now() - start > timeoutMs) {
//         console.error(`⏰ [Timeout] Gave up waiting after ${timeoutMs / 1000} seconds.`);
//         throw new Error(`Timeout while waiting for job '${jobName}' to complete.`);
//       }

//       await new Promise((resolve) => setTimeout(resolve, pollInterval));
//     }

//   } catch (error) {
//     console.error(`❌ [Fatal] waitForJobCompletion crashed: ${error.message}`);
//     throw error;
//   }
// }


// const runCommand = (cmd, cwd = process?.env?.DOCKER_FILE_DIR) => {
//   return new Promise((resolve, reject) => {
//     console.log(`🔹 Executing: ${cmd} (in ${cwd})`); // Log command & directory
//     exec(cmd, { cwd: cwd }, (error, stdout, stderr) => {
//       if (error) {
//         console.error(`❌ Command failed: ${cmd}\nError: ${error.message}`);
//         console.error(`Stderr:\n${stderr}`);
//         reject(new Error(stderr || error.message));
//         return;
//       }
//       console.log(`✅ Command succeeded:\n${stdout}`);
//       resolve(stdout.trim()); // Trim output for cleaner logs
//     });
//   });
// };


export async function waitForJobCompletion(
  jobName,
  namespace,
  timeoutMs = 600000, // 10 minutes
  pollInterval = 5000 // 5 seconds
) {
  console.log(`⏳ [waitForJobCompletion] Monitoring job: '${jobName}' in namespace: '${namespace}'`);

  const kc = new k8s.KubeConfig();
  const start = Date.now();

  try {
    kc.loadFromFile(loadPatchedMinikubeConfig());

    const k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);
    const k8sCoreApi = kc.makeApiClient(k8s.CoreV1Api);

    while (true) {
      try {
        const jobResp = await k8sBatchApi.readNamespacedJob(jobName, namespace);
        const jobStatus = jobResp.body?.status;

        console.log(`🔍 [Status] job=${jobName}, succeeded=${jobStatus?.succeeded || 0}, failed=${jobStatus?.failed || 0}`);

        // ✅ Check for explicit job success
        if (jobStatus?.succeeded === 1) {
          console.log(`✅ [Success] Job '${jobName}' completed.`);
        } else {
          // 🔄 Fallback: Check if pod phase is 'Succeeded'
          const podList = await k8sCoreApi.listNamespacedPod(
            namespace,
            undefined,
            undefined,
            undefined,
            undefined,
            `job-name=${jobName}`
          );

          const pod = podList.body.items[0];
          const podPhase = pod?.status?.phase;

          if (podPhase === "Succeeded") {
            console.log(`✅ [Pod Success] Pod '${pod.metadata.name}' has phase: ${podPhase}`);
          } else {
            console.log(`⏳ [Pod Pending] Pod '${pod?.metadata?.name || "?"}' is in phase: ${podPhase}`);
            if (Date.now() - start > timeoutMs) {
              throw new Error(`⏰ Timeout while waiting for job '${jobName}' to complete.`);
            }
            await new Promise((resolve) => setTimeout(resolve, pollInterval));
            continue; // Wait and retry
          }
        }

        // 📄 Fetch logs once job is considered completed
        const podList = await k8sCoreApi.listNamespacedPod(
          namespace,
          undefined,
          undefined,
          undefined,
          undefined,
          `job-name=${jobName}`
        );

        const podName = podList.body.items[0]?.metadata?.name;
        if (podName) {
          const logResp = await k8sCoreApi.readNamespacedPodLog(podName, namespace);
          console.log(`📄 [Pod Logs for ${podName}]:\n${logResp.body}`);
        } else {
          console.warn("⚠️ No pod found to fetch logs from.");
        }

        return true; // ✅ Done

      } catch (apiErr) {
        const statusCode = apiErr?.response?.statusCode;
        const rawBody = apiErr?.response?.body;

        if (statusCode === 404) {
          console.warn(`⚠️ [NotFound] Job '${jobName}' not found. It may not be created yet or was deleted.`);
        } else {
          console.error(`❌ [API Error] Failed to fetch job status. Code=${statusCode}`);
          if (rawBody) {
            try {
              const parsed = JSON.parse(rawBody);
              console.error(`📄 [ErrorBody]: ${JSON.stringify(parsed, null, 2)}`);
            } catch {
              console.error(`📄 [RawBody]: ${rawBody}`);
            }
          }
          throw apiErr;
        }
      }

      if (Date.now() - start > timeoutMs) {
        console.error(`⏰ [Timeout] Gave up waiting after ${timeoutMs / 1000} seconds.`);
        throw new Error(`Timeout while waiting for job '${jobName}' to complete.`);
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

  } catch (error) {
    console.error(`❌ [Fatal] waitForJobCompletion crashed: ${error.message}`);
    throw error;
  }
}


const runCommand = (cmd, cwd = process?.env?.DOCKER_FILE_DIR) => {
  return new Promise((resolve, reject) => {
    console.log(`🔹 [runCommand] Executing command:\n  → ${cmd}\n  → Working directory: ${cwd}`);

    exec(cmd, { cwd }, (error, stdout, stderr) => {
      if (error) {
        console.error(`❌ [runCommand] Command execution failed.`);
        console.error(`   ↳ Command   : ${cmd}`);
        console.error(`   ↳ Directory : ${cwd}`);
        console.error(`   ↳ Error     : ${error.message}`);
        if (stderr) console.error(`   ↳ Stderr    :\n${stderr}`);
        if (stdout) console.error(`   ↳ Stdout    :\n${stdout}`);

        // Include stderr and command in error message
        const errorMsg = `Command failed: "${cmd}" in "${cwd}"\nError: ${stderr || error.message}`;
        return reject(new Error(errorMsg));
      }

      console.log(`✅ [runCommand] Command executed successfully:\n  → ${cmd}`);
      if (stdout.trim()) console.log(`   ↳ Output:\n${stdout.trim()}`);
      resolve(stdout.trim());
    });
  });
};

const generateRandomString = (length = 4) => {
  return Array.from({ length: length }, () =>
    String.fromCharCode(97 + Math.floor(Math.random() * 26))
  ).join("");
};

// export async function sendDocketStatus(uuid, status) {
//   const broker = "54.251.96.179:9092";
//   const topic = "docket-status";
 
//   // Create topic if it doesn't exist
//   allfunction.createTopicIfNotExists(broker, topic);
 
//   const kafka = new Kafka({ brokers: [broker] });
//   const producer = kafka.producer();
//   await producer.connect();
 
//   const message = {
//     uuid,
//     status
//   };
 
//   await producer.send({
//     topic,
//     messages: [{
//       key: uuid,
//       value: JSON.stringify(message)
//     }]
//   });
 
//   console.log("Message sent to topic 'docket-status':", message);
 
//   await producer.disconnect();
  
// }

export async function sendDocketStatus(uuid, status) {
  const broker = "54.251.96.179:9092";
  const topic = "docket-status";

  try {
    console.log(`📡 [sendDocketStatus] Sending status '${status}' for UUID: ${uuid}`);

    await allfunction.createTopicIfNotExists(broker, topic);

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

    console.log(`✅ [sendDocketStatus] Message sent to topic '${topic}':`, message);
    await producer.disconnect();
  } catch (error) {
    console.error(`❌ [sendDocketStatus] Failed to send Kafka message: ${error.message}`);
    throw error;
  }
}

function loadPatchedMinikubeConfig() {
  const originalPath = '/app/.kube/config';
  const raw = fs.readFileSync(originalPath, 'utf8');
  const config = yaml.load(raw);

  const patchPath = (p) => p?.replace('/home/ubuntu/.minikube', '/app/.minikube'); 
  config.clusters?.forEach(c => {
    if (c.cluster['certificate-authority']) {
      c.cluster['certificate-authority'] = patchPath(c.cluster['certificate-authority']);
    }
 
    // ✅ Patch the Kubernetes API server address

  //   if (c.cluster['server']) {
  //     c.cluster['server'] = c.cluster['server'].replace(
  //       'https://192.168.49.2',
  //       'https://host.docker.internal'
  //     );
  //   }
  });
 
  config.users?.forEach(u => {

    if (u.user['client-certificate']) {
      u.user['client-certificate'] = patchPath(u.user['client-certificate']);
    }
    if (u.user['client-key']) {
      u.user['client-key'] = patchPath(u.user['client-key']);
    }
  });

  const tmpPath = '/tmp/kubeconfig-patched.yaml';
  fs.writeFileSync(tmpPath, yaml.dump(config));
  return tmpPath;
}


/**
 * Set Minikube Docker environment and remove old containers/images
 */
export async function cleanMinikubeDockerResources() {
  try {
    console.log("🔌 [cleanup] Connecting to Minikube Docker daemon...");

    // Step 1: Load docker-env vars from minikube
    const dockerEnvRaw = exec("minikube docker-env --shell bash").toString();
    const lines = dockerEnvRaw.split("\n");

    const minikubeEnv = {};
    for (const line of lines) {
      if (line.startsWith("export ")) {
        const [key, val] = line.replace("export ", "").split("=", 2);
        minikubeEnv[key] = val.replace(/"/g, "");
      }
    }
    console.log("minikubeEnv is *** : ", minikubeEnv)
    console.log("dockerEnvRaw is *** :", dockerEnvRaw)
    // Step 2: Setup Docker client using minikube env
    const docker = new Docker({
      host: minikubeEnv.DOCKER_HOST?.replace("tcp", "http"),
      port: 2376,
      ca: path.join(minikubeEnv.DOCKER_CERT_PATH, "ca.pem"),
      cert: path.join(minikubeEnv.DOCKER_CERT_PATH, "cert.pem"),
      key: path.join(minikubeEnv.DOCKER_CERT_PATH, "key.pem"),
    });

    console.log("✅ [cleanup] Connected to Minikube Docker");

    // Step 3: Remove containers using image
    const containers = await docker.listContainers({ all: true, filters: { ancestor: ["nagagogulan/aimx-evaluation:latest"] } });
    for (const c of containers) {
      const container = docker.getContainer(c.Id);
      console.log(`🛑 [cleanup] Removing container: ${c.Id}`);
      try {
        await container.stop();
      } catch {}
      await container.remove({ force: true });
    }

    // Step 4: Remove image
    const images = await docker.listImages({ filters: { reference: ["nagagogulan/aimx-evaluation:latest"] } });
    for (const img of images) {
      const image = docker.getImage(img.Id);
      console.log(`🧼 [cleanup] Removing image: ${img.Id}`);
      await image.remove({ force: true });
    }

    console.log("🚮 [cleanup] Done removing containers + image");

  } catch (err) {
    console.error(`❌ [cleanup] Docker cleanup failed: ${err.message}`);
  }
}


 
