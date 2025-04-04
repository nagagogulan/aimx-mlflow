import { exec } from "child_process";
import { nanoid } from "nanoid";

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
