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
import { glob } from 'glob';


const MLFLOW_API_BASE = "http://54.251.96.179:5000/api/2.0/mlflow";
const projectRoot = "/app"; // ✅ Container-based 
console.log('PROJECT ROOT:', projectRoot);

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
    const ENTRYPOINT_SH_FILE = `${INFERENCE_BASE_DIR}/entrypoint.sh`;

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

    await runCommand(`cp ${ENTRYPOINT_SH_FILE} ${TARGET_DIR}`);
    await runCommand(`chmod +x ${TARGET_DIR}/entrypoint.sh`);

    // Unzip model ZIP archive
    console.log(`[${tempId}] Unzipping model archive from: ${modelZipPath}`);
    await new Promise((resolve, reject) => {
      fs.createReadStream(modelZipPath)
        .pipe(unzipper.Extract({ path: TEMP_UNZIP_DIR }))
        .on('close', resolve)
        .on('error', reject);
    });

    let weightFileName = null;

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
      } else if (fileName.match(/\.(h5|joblib|pkl|onnx|pth|bin)$/)) {
        console.log(`[${tempId}] Moving model weight: ${fileName}`);
        await runCommand(`mv "${filePath}" ${MODEL_WEIGHT_DIR}/${fileName}`);
        weightFileName = fileName;

      } else {
        console.log(`[${tempId}] Moving other file: ${fileName}`);
        await runCommand(`mv "${filePath}" ${MODEL_WEIGHT_DIR}/${fileName}`);
      }
    }

    // Clean up unzip folder
    await runCommand(`rm -rf ${TEMP_UNZIP_DIR}`);

    // Copy dataset file
    let datasetPayloadPath;

    if (datasetFileName.endsWith('.zip')) {
      console.log(`[${tempId}] Unzipping dataset ZIP: ${datasetFileName}`);
      await new Promise((resolve, reject) => {
        fs.createReadStream(datasetFullPath)
          .pipe(unzipper.Extract({ path: DATASETS_DIR }))
          .on('close', resolve)
          .on('error', reject);
      });
      // Flatten if needed: check if only one folder was created
      const unzippedItems = fs.readdirSync(DATASETS_DIR);
      if (unzippedItems.length === 1) {
        const singleFolder = path.join(DATASETS_DIR, unzippedItems[0]);
        if (fs.statSync(singleFolder).isDirectory()) {
          console.log(`[${tempId}] Flattening extra folder layer: ${singleFolder}`);
          await runCommand(`mv "${singleFolder}"/* "${DATASETS_DIR}/"`);
          await runCommand(`rm -rf "${singleFolder}"`);
        }
      }
      datasetPayloadPath = './datasets/';
    } else {
      console.log(`[${tempId}] Copying dataset file: ${datasetFileName}`);
      await runCommand(`cp "${datasetFullPath}" "${DATASETS_DIR}/${datasetFileName}"`);
      datasetPayloadPath = `./datasets/${datasetFileName}`;
    }



    const payload = {
      tempId,
      targetDir: TARGET_DIR,
      weightsPath: weightFileName ? `./weights/${weightFileName}` : `./weights/`,
      datasetPath: datasetPayloadPath,
      tempReq: `./temporal-runs/${tempId}/datasets/${datasetFileName}`,
    };

    // Optional image classification flow
    // if (
    //   dataType === "unstructured" &&
    //   taskType === "image-classification" &&
    //   modelFramework === "onnx" &&
    //   modelArchitecture === "resnet"
    // ) {
    //   const imageZipFileName = path.basename(new URL(imageZipUrl).pathname);
    //   const dataLabelFileName = path.basename(new URL(dataLabelUrl).pathname);

    //   await runCommand(`cp ${DATASETS_DIR}/${imageZipFileName} ${imageZipUrl}`);
    //   await runCommand(`cp ${DATASETS_DIR}/${dataLabelFileName} ${dataLabelUrl}`);

    //   payload.imageZipPath = `./datasets/${imageZipFileName}`;
    //   payload.dataLabelPath = `./datasets/${dataLabelFileName}`;
    // }

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

export async function runEvaluationsInCluster(options, inferenceData) {
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

async function getContainerEnvConfig(options, inferenceData) {

  console.log(`📦 [getContainerEnvConfig] Generating container environment configuration for options:`, options, inferenceData);
  const baseEnv = [
    { name: "MODEL_WEIGHTS_PATH", value: inferenceData.weightsPath },
    { name: "MLFLOW_TRACKING_URI", value: process.env.MLFLOW_URL },
    { name: "DATASET_PATH", value: inferenceData.datasetPath },
    { name: "EXPERIMENT_NAME", value: options.uuid },
  ];

  if(!!options.target_column) {
    baseEnv.push({ name: "TARGET_COLUMN", value: options.target_column });
  }

  console.log("final base env:********************", baseEnv)
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

