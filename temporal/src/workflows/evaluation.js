import { proxyActivities } from "@temporalio/workflow";

const {
  helloWorld,
  copyInferenceScripts,
  buildDockerImage,
  runEvaluations,
  runEvaluationsInCluster,
  waitForJobCompletion,
  sendDocketStatus,
} = proxyActivities({
  startToCloseTimeout: "5 minute",
  retry: {
    initialInterval: "60 second",
    maximumAttempts: 1,
    // maximumAttempts: 5,
    // backoffCoefficient: 2,
  },
});

export async function runEval(payload) {
  const inferenceData = await copyInferenceScripts(payload);
  const buildData = await buildDockerImage(inferenceData);
  const evalData = await runEvaluationsInCluster(payload, inferenceData);
  const jobStatus = await waitForJobCompletion(
    evalData.jobName,
    evalData.namespace
  );
  // const evalData = await runEvaluations(inferenceData);  
  console.log("evalDat is ", jobStatus);
  if (jobStatus) {
    
  const uuidFromProcessedData = payload.uuid; // Replace with real uuid
  const status = "success"; // or "failed"
 
await sendDocketStatus(uuidFromProcessedData, status)
    .catch(err => {
      console.error("Failed to send Kafka message:", err);
    });
  }

  return {
    status: "OK",
    inferenceData: inferenceData,
    buildData: buildData,
    evalData: evalData,
    jobStatus: jobStatus,
  };
}
