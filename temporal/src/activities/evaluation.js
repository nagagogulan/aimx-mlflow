import { exec } from "child_process";
import { nanoid } from "nanoid";
import * as k8s from "@kubernetes/client-node";

export async function helloWorld(options) {
  return "hello world";
}

export async function copyInferenceScripts(options) {
  const tempId = nanoid();
  const INFERENCE_BASE_DIR = `${process?.env?.EVAL_BASE_DIR}/scripts/text-classification/distilbert/pkl`;
  const INFERENCE_SCRIPT_PATH = `${INFERENCE_BASE_DIR}/src`;
  const REQUIREMENTS_FILE = `${INFERENCE_BASE_DIR}/requirements.txt`;
  const DOCKER_FILE_DIR = `${INFERENCE_BASE_DIR}/Dockerfile`;

  const TARGET_DIR = `${process?.env?.EVAL_BASE_DIR}/temporal-runs/${tempId}`;
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

export async function runEvaluationsInCluster(options) {
  const kc = new k8s.KubeConfig();
  kc.loadFromDefault(); // This will load from ~/.kube/config
  const k8sApi = kc.makeApiClient(k8s.CoreV1Api);
  const k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);

  // generate random string that returns 4 characters (only alphabets)
  const randomString = generateRandomString(4).toLocaleLowerCase();

  const namespace = options?.namespace || "default";
  const jobName = `aimx-evaluation-${randomString}`; // Unique job name

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
          containers: [
            {
              name: "aimx-evaluation",
              image: "aimx-evaluation:latest",
              imagePullPolicy: "Never", // Use local image
              env: [
                {
                  name: "MODEL_WIGHTS_PATH",
                  value: process.env.WEIGHTS_PATH,
                },
                {
                  name: "MLFLOW_TRACKING_URI",
                  value: process.env.MLFLOW_URL,
                },
              ],
            },
          ],
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
