import { exec } from "child_process";
import { nanoid } from "nanoid";
import * as k8s from "@kubernetes/client-node";
import path from "path";
// import path from 'path';
import unzipper from 'unzipper';
import { Kafka } from "kafkajs";
import * as allfunction from "../kafka/worker.js" ;
import fs from 'fs';
import https from 'https';
import yaml from 'js-yaml';
import axios from "axios";
import { promises as fsp } from 'fs';


const MLFLOW_API_BASE = "http://54.251.96.179:5000/api/2.0/mlflow";
const projectRoot = "/app"; // ✅ Container-based 
console.log('PROJECT ROOT:', projectRoot);

export async function helloWorld(options) {
  return "hello world";
}


// export async function copyInferenceScripts(options) {
//   const tempId = nanoid();
//   console.log(`📥 [copyInferenceScripts] Received options for ID: ${tempId}`);

//   try {
//     const {
//       dataType,
//       taskType,
//       modelFramework,                                                                                                                                                                                                                                                                                                                                                                                                                                 
//       modelArchitecture,
//       modelWeightUrl,
//       modelDatasetUrl,
//       imageZipUrl,
//       dataLabelUrl,
//     } = options;

//     if (!modelWeightUrl?.link && !modelWeightUrl?.path) {
//       console.error("Missing 'modelWeightUrl.link' or 'modelWeightUrl.path' in input options");
//       throw new Error("Missing required field: modelWeightUrl.link or modelWeightUrl.path");
//     }

//     const datasetEntry = modelDatasetUrl?.[0];
//     if (!datasetEntry?.Value) {
//       console.error("Missing 'modelDatasetUrl[0].Value' in input options");
//       throw new Error("Missing required field: modelDatasetUrl[0].Value");
//     }

//     const INFERENCE_BASE_DIR = `${projectRoot}/scripts/${dataType}/${taskType}/${modelFramework}/${modelArchitecture}`;
//     const INFERENCE_SCRIPT_PATH = `${INFERENCE_BASE_DIR}/src`;
//     const REQUIREMENTS_FILE = `${INFERENCE_BASE_DIR}/requirements.txt`;
//     const DOCKER_FILE_DIR = `${INFERENCE_BASE_DIR}/Dockerfile`;

//     const TARGET_DIR = `${projectRoot}/temporal-runs/${tempId}`;
//     const MODEL_WEIGHT_DIR = `${TARGET_DIR}/weights`;
//     const DATASETS_DIR = `${TARGET_DIR}/datasets`;

//     const modelFileName = modelWeightUrl.link
//       ? path.basename(new URL(modelWeightUrl.link).pathname)
//       : path.basename(modelWeightUrl.path);
//     const datasetFileName = path.basename(datasetEntry.Value);

//     const modelWeightFullPath = modelWeightUrl.path
//       ? path.resolve(modelWeightUrl.path)
//       : null;
//     const datasetFullPath = path.resolve(datasetEntry.Value);

//     console.log(`[${tempId}] Creating target directory: ${TARGET_DIR}`);
//     await runCommand(`mkdir -p ${TARGET_DIR}`);

//     console.log(`[${tempId}] Copying inference scripts from ${INFERENCE_SCRIPT_PATH}`);
//     await runCommand(`cp -r ${INFERENCE_SCRIPT_PATH} ${TARGET_DIR}`);

//     console.log(`[${tempId}] Copying Dockerfile`);
//     await runCommand(`cp ${DOCKER_FILE_DIR} ${TARGET_DIR}`);

//     console.log(`[${tempId}] Creating weights directory`);
//     await runCommand(`mkdir -p ${MODEL_WEIGHT_DIR}`);

//     if (modelWeightUrl.type === "GIT" && modelWeightUrl.link) {
//       console.log(`[${tempId}] Downloading model weight from GitHub: ${modelWeightUrl.link}`);
//       if (!modelWeightUrl.pat) {
//         throw new Error("GitHub PAT is missing in 'modelWeightUrl.pat'.");
//       }

//       const headers = {
//         Authorization: `token ${modelWeightUrl.pat}`,
//         Accept: "application/vnd.github.v3.raw",
//       };

//       const response = await axios.get(modelWeightUrl.link, { headers, responseType: "stream" });
//       const modelWeightFilePath = `${MODEL_WEIGHT_DIR}/${modelFileName}`;
//       const writer = fs.createWriteStream(modelWeightFilePath);

//       response.data.pipe(writer);

//       await new Promise((resolve, reject) => {
//         writer.on("finish", resolve);
//         writer.on("error", reject);
//       });

//       console.log(`[${tempId}] Model weight downloaded to: ${modelWeightFilePath}`);

//       // Copy the downloaded file to the temporal-runs directory
//       console.log(`[${tempId}] Copying model weight to temporal-runs: ${MODEL_WEIGHT_DIR}/${modelFileName}`);
//       await runCommand(`cp ${modelWeightFilePath} ${MODEL_WEIGHT_DIR}/${modelFileName}`);
//     // } else if (modelWeightUrl.path) {
//     //   console.log(`[${tempId}] Copying model weight: ${modelFileName}`);
//     //   await runCommand(`cp ${modelWeightFullPath} ${MODEL_WEIGHT_DIR}/${modelFileName}`);
//     //   // If the file is a zip, extract and organize contents
//     //   if (modelFileName.endsWith('.zip')) {
//     //     const TEMP_UNZIP_DIR = `${TARGET_DIR}/unzipped`;
//     //     console.log(`[${tempId}] Detected zip file. Extracting to: ${TEMP_UNZIP_DIR}`);
//     //     await runCommand(`mkdir -p ${TEMP_UNZIP_DIR}`);
//     //     await runCommand(`unzip -o ${MODEL_WEIGHT_DIR}/${modelFileName} -d ${TEMP_UNZIP_DIR}`);
//     //     const files = fs.readdirSync(TEMP_UNZIP_DIR);
//     //     // Move all .txt files as requirements.txt (if multiple, log and move all)
//     //     const txtFiles = files.filter(f => f.endsWith('.txt'));
//     //     if (txtFiles.length > 0) {
//     //       for (const txtFile of txtFiles) {
//     //         console.log(`[${tempId}] Moving requirements file (${txtFile}) to run root.`);
//     //         await runCommand(`mv ${TEMP_UNZIP_DIR}/${txtFile} ${TARGET_DIR}/${txtFile}`);
//     //       }
//     //     } else {
//     //       console.log(`[${tempId}] No .txt requirements file found in zip.`);
//     //     }
//     //     // Move all .py files to src/ as evaluation scripts
//     //     const srcDir = `${TARGET_DIR}/src`;
//     //     await runCommand(`mkdir -p ${srcDir}`);
//     //     const pyFiles = files.filter(f => f.endsWith('.py'));
//     //     if (pyFiles.length > 0) {
//     //       for (const pyFile of pyFiles) {
//     //         console.log(`[${tempId}] Moving evaluation script (${pyFile}) to src/ folder.`);
//     //         await runCommand(`mv ${TEMP_UNZIP_DIR}/${pyFile} ${srcDir}/${pyFile}`);
//     //       }
//     //     } else {
//     //       console.log(`[${tempId}] No Python evaluation script found in zip.`);
//     //     }
//     //     // Move any other file as the model file to weights/
//     //     const handled = new Set([...txtFiles, ...pyFiles]);
//     //     const otherFiles = files.filter(f => !handled.has(f));
//     //     if (otherFiles.length > 0) {
//     //       for (const otherFile of otherFiles) {
//     //         console.log(`[${tempId}] Moving model file (${otherFile}) to weights/ folder.`);
//     //         await runCommand(`mv ${TEMP_UNZIP_DIR}/${otherFile} ${MODEL_WEIGHT_DIR}/${otherFile}`);
//     //       }
//     //     } else {
//     //       console.log(`[${tempId}] No model file found in zip (non .py/.txt).`);
//     //     }
//     //     // Clean up unzip dir
//     //     console.log(`[${tempId}] Cleaning up temporary unzip directory.`);
//     //     await runCommand(`rm -rf ${TEMP_UNZIP_DIR}`);
//     //   }
//     // } else {
//     //   throw new Error("Invalid modelWeightUrl format.");
//     // }
//     } else if (modelWeightUrl.path) {
//         console.log(`[${tempId}] Copying model weight: ${modelFileName}`);
//         await runCommand(`cp ${modelWeightFullPath} ${MODEL_WEIGHT_DIR}/${modelFileName}`);

//         if (modelFileName.endsWith('.zip')) {
//           const TEMP_UNZIP_DIR = `${TARGET_DIR}/unzipped`;
//           console.log(`[${tempId}] Detected zip file. Extracting to: ${TEMP_UNZIP_DIR}`);
//           await runCommand(`mkdir -p ${TEMP_UNZIP_DIR}`);

//           // Unzip using unzipper
//           await new Promise((resolve, reject) => {
//             fs.createReadStream(`${MODEL_WEIGHT_DIR}/${modelFileName}`)
//               .pipe(unzipper.Extract({ path: TEMP_UNZIP_DIR }))
//               .on('close', resolve)
//               .on('error', reject);
//           });

//           console.log(`[${tempId}] Successfully unzipped using unzipper`);

//           const files = await fsp.readdir(TEMP_UNZIP_DIR);
//           const srcDir = `${TARGET_DIR}/src`;
//           await runCommand(`mkdir -p ${srcDir}`);

//           const evalFiles = files.filter(f => f.endsWith('.py') || f.endsWith('.ipynb'));
//           for (const evalFile of evalFiles) {
//             console.log(`[${tempId}] Moving evaluation script (${evalFile}) to src/ folder.`);
//             await runCommand(`mv ${TEMP_UNZIP_DIR}/${evalFile} ${srcDir}/${evalFile}`);
//           }

//           const txtFiles = files.filter(f => f.endsWith('.txt'));
//           for (const txtFile of txtFiles) {
//             console.log(`[${tempId}] Moving requirements file (${txtFile}) to run root.`);
//             await runCommand(`mv ${TEMP_UNZIP_DIR}/${txtFile} ${TARGET_DIR}/${txtFile}`);
//           }

//           const handled = new Set([...txtFiles, ...evalFiles]);
//           const otherFiles = files.filter(f => !handled.has(f));
//           for (const otherFile of otherFiles) {
//             console.log(`[${tempId}] Moving model file (${otherFile}) to weights/ folder.`);
//             await runCommand(`mv ${TEMP_UNZIP_DIR}/${otherFile} ${MODEL_WEIGHT_DIR}/${otherFile}`);
//           }

//           console.log(`[${tempId}] Cleaning up temporary unzip directory.`);
//           await runCommand(`rm -rf ${TEMP_UNZIP_DIR}`);
//         }
//     } else {
//       throw new Error("Invalid modelWeightUrl format.");
//     }


//     console.log(`[${tempId}] Creating datasets directory`);
//     await runCommand(`mkdir -p ${DATASETS_DIR}`);

//     console.log(`[${tempId}] Copying dataset: ${datasetFileName}`);
//     await runCommand(`cp ${datasetFullPath} ${DATASETS_DIR}/${datasetFileName}`);

//     console.log(`[${tempId}] Copying requirements.txt`);
//     await runCommand(`cp ${REQUIREMENTS_FILE} ${TARGET_DIR}`);

//     const payload = {
//       tempId,
//       targetDir: TARGET_DIR,
//       weightsPath: `./weights/${modelFileName}`,
//       datasetPath: `./datasets/${datasetFileName}`,
//       tempReq: `./temporal-runs/${tempId}/datasets/${datasetFileName}`, //toget target column this pat is used
//     };

//     if (
//       dataType === "unstructured" &&
//       taskType === "image-classification" &&
//       modelFramework === "onnx" &&
//       modelArchitecture === "resnet"
//     ) {
//       const imageZipFileName = path.basename(new URL(imageZipUrl).pathname);
//       const dataLabelFileName = path.basename(new URL(dataLabelUrl).pathname);

//       console.log(`[${tempId}] Copying image zip: ${imageZipFileName}`);
//       await runCommand(`cp ${DATASETS_DIR}/${imageZipFileName} ${imageZipUrl}`);

//       console.log(`[${tempId}] Copying label file: ${dataLabelFileName}`);
//       await runCommand(`cp ${DATASETS_DIR}/${dataLabelFileName} ${dataLabelUrl}`);

//       payload.imageZipPath = `./datasets/${imageZipFileName}`;
//       payload.dataLabelPath = `./datasets/${dataLabelFileName}`;
      
//     }

//     console.log(`✅ [${tempId}] All files copied successfully.`);

//     return payload
//   } catch (err) {
//     console.error(`❌ [${tempId}] Failed in copyInferenceScripts:`, err.message);
//     throw err;
//   }
// }

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

    const datasetEntry = modelDatasetUrl?.[0];
    if (!datasetEntry?.Value) {
      throw new Error("Missing required field: modelDatasetUrl[0].Value");
    }

    const INFERENCE_BASE_DIR = `${projectRoot}/scripts/`;
    const DOCKER_FILE_DIR = `${INFERENCE_BASE_DIR}/Dockerfile`;

    const TARGET_DIR = `${projectRoot}/temporal-runs/${tempId}`;
    const MODEL_WEIGHT_DIR = `${TARGET_DIR}/weights`;
    const SRC_DIR = `${TARGET_DIR}/src`;
    const DATASETS_DIR = `${TARGET_DIR}/datasets`;
    const TEMP_UNZIP_DIR = `${TARGET_DIR}/unzipped`;

    const datasetFileName = path.basename(datasetEntry.Value);
    const datasetFullPath = path.resolve(datasetEntry.Value);

    const modelZipPath = path.resolve(modelWeightUrl.path);
    const modelZipFileName = path.basename(modelZipPath);

    // Create necessary directories
    await runCommand(`mkdir -p ${TARGET_DIR}`);
    await runCommand(`mkdir -p ${MODEL_WEIGHT_DIR}`);
    await runCommand(`mkdir -p ${SRC_DIR}`);
    await runCommand(`mkdir -p ${DATASETS_DIR}`);
    await runCommand(`mkdir -p ${TEMP_UNZIP_DIR}`);

    // Copy Dockerfile
    console.log(`[${tempId}] Copying Dockerfile from: ${DOCKER_FILE_DIR}`);
    await runCommand(`cp ${DOCKER_FILE_DIR} ${TARGET_DIR}`);

    // Unzip model ZIP archive
    console.log(`[${tempId}] Unzipping model archive from: ${modelZipPath}`);
    await new Promise((resolve, reject) => {
      fs.createReadStream(modelZipPath)
        .pipe(unzipper.Extract({ path: TEMP_UNZIP_DIR }))
        .on('close', resolve)
        .on('error', reject);
    });

    // Recursively find all files in unzip dir
    const files = glob.sync(`${TEMP_UNZIP_DIR}/**/*`, { nodir: true });

    for (const filePath of files) {
      const fileName = path.basename(filePath);

      if (fileName.endsWith('.py') || fileName.endsWith('.ipynb')) {
        console.log(`[${tempId}] Moving eval file: ${fileName}`);
        await runCommand(`mv "${filePath}" ${SRC_DIR}/${fileName}`);
      } else if (fileName.endsWith('.txt')) {
        console.log(`[${tempId}] Moving requirements file: ${fileName}`);
        await runCommand(`mv "${filePath}" ${TARGET_DIR}/${fileName}`);
      } else if (fileName.match(/\.(pkl|onnx|pt|bin)$/)) {
        console.log(`[${tempId}] Moving model weight: ${fileName}`);
        await runCommand(`mv "${filePath}" ${MODEL_WEIGHT_DIR}/${fileName}`);
      } else {
        console.log(`[${tempId}] Moving other file: ${fileName}`);
        await runCommand(`mv "${filePath}" ${MODEL_WEIGHT_DIR}/${fileName}`);
      }
    }

    // Clean up unzip folder
    await runCommand(`rm -rf ${TEMP_UNZIP_DIR}`);

    // Copy dataset file
    console.log(`[${tempId}] Copying dataset: ${datasetFileName}`);
    await runCommand(`cp ${datasetFullPath} ${DATASETS_DIR}/${datasetFileName}`);

    const payload = {
      tempId,
      targetDir: TARGET_DIR,
      weightsPath: `./weights/`,
      datasetPath: `./datasets/${datasetFileName}`,
      tempReq: `./temporal-runs/${tempId}/datasets/${datasetFileName}`,
    };

    // Optional image classification flow
    if (
      dataType === "unstructured" &&
      taskType === "image-classification" &&
      modelFramework === "onnx" &&
      modelArchitecture === "resnet"
    ) {
      const imageZipFileName = path.basename(new URL(imageZipUrl).pathname);
      const dataLabelFileName = path.basename(new URL(dataLabelUrl).pathname);

      await runCommand(`cp ${DATASETS_DIR}/${imageZipFileName} ${imageZipUrl}`);
      await runCommand(`cp ${DATASETS_DIR}/${dataLabelFileName} ${dataLabelUrl}`);

      payload.imageZipPath = `./datasets/${imageZipFileName}`;
      payload.dataLabelPath = `./datasets/${dataLabelFileName}`;
    }

    console.log(`✅ [${tempId}] Inference preparation complete.`);
    return payload;

  } catch (err) {
    console.error(`❌ [${tempId}] Error in copyInferenceScripts:`, err.message);
    throw err;
  }
}



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
    await runCommand("docker build --no-cache -t aimx-evaluation .", dir);
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

  // await cleanMinikubeDockerResources();
  const randomString = generateRandomString(4).toLowerCase();
  const namespace = process?.env?.NAMESPACE || "default";
  if (!namespace || typeof namespace !== "string") {
    throw new Error("❌ Invalid namespace: " + JSON.stringify(namespace));
  }
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

    const containerData = await getContainerEnvConfig(options, inferenceData);

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

    const test = await k8sBatchApi.createNamespacedJob({ namespace, body: jobManifest });
    console.log("✅ Job response received:", JSON.stringify(test.body || test, null, 2));

    console.log(`✅ Kubernetes job '${jobName}' created successfully.`);
    return {
      jobName: jobName.trim(),
      namespace: namespace.trim(),
     };
  } catch (error) {
    console.error(`❌ Failed to create Kubernetes job '${jobName}':`, error.message);
    throw error;
  }
}

function getTargetColumnFromCSV(csvPath) {
  const absPath = path.resolve(csvPath);
  console.log(`[getTargetColumnFromCSV] Reading CSV file at: ${absPath}`);
  const firstLine = fs.readFileSync(absPath, 'utf8').split('\n')[0];
  console.log(`[getTargetColumnFromCSV] First line (header): ${firstLine}`);
  const columns = firstLine.split(',').map(col => col.trim());
  console.log(`[getTargetColumnFromCSV] Columns found: ${columns.join(', ')}`);
  const targetColumn = columns[columns.length - 1];
  console.log(`[getTargetColumnFromCSV] Selected target column: ${targetColumn}`);
  return targetColumn;
}

async function getContainerEnvConfig(options, inferenceData) {
  let targetColumn = "target";
  if(options.modelFramework != "tensorflow"){
    targetColumn = await getTargetColumnFromCSV(inferenceData.tempReq);
  }
  console.log(`📦 [getContainerEnvConfig] Generating container environment configuration for options:`, options, inferenceData);
  const baseEnv = [
    { name: "MODEL_WIGHTS_PATH", value: inferenceData.weightsPath },
    { name: "MLFLOW_TRACKING_URI", value: process.env.MLFLOW_URL },
    { name: "DATASET_PATH", value: inferenceData.datasetPath },
    { name: "EXPERIMENT_NAME", value: options.uuid },
    { name: "TARGET_COLUMN", value: targetColumn ? targetColumn : 'target'}
  ];
  console.log(`📦 [getContainerEnvConfig] Base environment variables:`, baseEnv);

  if (
    options.dataType === "unstructured" &&
    options.taskType === "image-classification" &&
    options.modelFramework === "onnx"
  ) {
    baseEnv.push(
      { name: "IMAGES_ZIP_PATH", value: inferenceData.imageZipPath },
      { name: "DATA_LABEL_PATH", value: inferenceData.dataLabelPath }
    );
  }

  return [{
    name: "aimx-evaluation",
    image: "nagagogulan/aimx-evaluation:latest",
    imagePullPolicy: "Always",
    env: baseEnv,
    workingDir: "/app"
  }];
}



export async function waitForJobCompletion(
  jobName,
  namespace,
  timeoutMs = 600000,
  pollInterval = 5000
) {
  const kc = new k8s.KubeConfig();
  // kc.loadFromDefault(); // This will load from ~/.kube/config
    kc.loadFromFile(loadPatchedMinikubeConfig()); // ✅ Use patched kubeconfig
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

export async function sendDocketMessage( uuid, status, metrics, publishtopic , payload = null ) {
  if(!uuid || !status || !metrics || !publishtopic) {
    return `Invalid input details: ${uuid} or ${status} or ${metrics} or ${publishtopic}`
  }

  console.log("recieved topic looks like:", publishtopic)
  const broker = "54.251.96.179:9092";
  const topic = publishtopic;
  
  try {
    console.log(`📡 [sendDocketMessage] Sending status '${status}' for UUID: ${uuid}`);

    await allfunction.createTopicIfNotExists(broker, topic);

    const kafka = new Kafka({ brokers: [broker] });
    const producer = kafka.producer();
    await producer.connect();

    const message = {
      uuid,
      status,
      metrics
    };

    if (payload) {
      message.payload = payload;
    }

    await producer.send({
      topic,
      messages: [{
        key: uuid,
        value: JSON.stringify(message),
      }]
    });

    console.log(`✅ [sendDocketMessage] Message sent to topic '${topic}':`, message);
    await producer.disconnect();
  } catch (error) {
    console.error(`❌ [sendDocketMessage] Failed to send Kafka message: ${error.message}`);
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

// activities/fetchJobMetrics.ts
export async function fetchJobMetrics(job){
  try {
    const experimentResponse = await getExperimentByName(job.uuid);

    if (!experimentResponse?.experiment?.experiment_id) {
      throw new Error("Experiment not found");
    }

    const experimentId = experimentResponse.experiment.experiment_id;

    const runsData = await fetchMlflowRuns(experimentId);
    const latestRunId = runsData?.runs?.[0]?.info?.run_id;

    if (!latestRunId) {
      throw new Error("No MLflow runs found for experiment");
    }

    const metrics = await getMlflowRunById(latestRunId);
    return metrics;
  } catch (error) {
    console.error(`❌ Failed to fetch metrics for job ${job?.uuid}:`, error?.message || error);
    throw new Error("Could not fetch MLflow metrics");
  }
}

async function getExperimentByName(name) {
  try {
    const url = `${MLFLOW_API_BASE}/experiments/get-by-name`;
    const response = await axios.get(url, {
      params: { experiment_name: name },
    });
    return response.data;
  } catch (error) {
    console.error(`❌ Error fetching experiment by name "${name}":`, error?.message || error);
    throw new Error("Failed to get experiment by name");
  }
}

async function fetchMlflowRuns(experimentId) {
  try {
    const url = `${MLFLOW_API_BASE}/runs/search`;
    const response = await axios.post(
      url,
      {
        experiment_ids: [experimentId],
        max_results: 10,
      },
      {
        headers: { "Content-Type": "application/json" },
      }
    );
    return response.data;
  } catch (error) {
    console.error(`❌ Error fetching MLflow runs for experiment ${experimentId}:`, error?.message || error);
    throw new Error("Failed to fetch MLflow runs");
  }
}

async function getMlflowRunById(runId) {
  try {
    const url = `${MLFLOW_API_BASE}/runs/get`;
    const response = await axios.get(url, {
      params: { run_id: runId },
    });
    return response.data;
  } catch (error) {
    console.error(`❌ Error fetching MLflow run by ID: ${runId}`, error?.message || error);
    throw new Error(`Failed to fetch run metrics for run ID: ${runId}`);
  }
}

