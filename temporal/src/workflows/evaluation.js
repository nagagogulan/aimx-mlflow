import { proxyActivities } from "@temporalio/workflow";

const { helloWorld, copyInferenceScripts, buildDockerImage, runEvaluations } =
  proxyActivities({
    startToCloseTimeout: "5 minute",
    retry: {
      initialInterval: "60 second",
      maximumAttempts: 5,
      backoffCoefficient: 2,
    },
  });

export async function runEval(payload) {
  const inferenceData = await copyInferenceScripts(payload);
  const buildData = await buildDockerImage(inferenceData);
  const evalData = await runEvaluations(inferenceData);
  return {
    status: "OK",
    inferenceData: inferenceData,
    buildData: buildData,
    evalData: evalData,
  };
}
