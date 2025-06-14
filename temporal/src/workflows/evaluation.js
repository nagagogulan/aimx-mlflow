import { proxyActivities } from "@temporalio/workflow";

const {
  helloWorld,
  copyInferenceScripts,
  buildDockerImage,
  runEvaluations,
  runEvaluationsInCluster,
  waitForJobCompletion,
  fetchJobMetrics,
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
  // if (jobStatus) {
  //   const metrics = await fetchJobMetrics(payload)
  //   const uuidFromProcessedData = payload.uuid; // Replace with real uuid
  //   const status = "success"; // or "failed"

  //   await sendDocketStatus(uuidFromProcessedData, status, metrics)
  //     .catch(err => {
  //       console.error("Failed to send Kafka message:", err);
  //     });
  // } else {
  //   const metrics = "Nil"
  //   const uuidFromProcessedData = payload.uuid; // Replace with real uuid
  //   const status = "Failed"; // or "failed"

  //   await sendDocketStatus(uuidFromProcessedData, status, metrics)
  //     .catch(err => {
  //       console.error("Failed to send Kafka message:", err);
  //     });
  // }

  const uuid = payload.uuid || "unknown-uuid";
  let status = "Failed";
  let metrics = "Nil";

  if (jobStatus) {
    status = "success";
    try {
      metrics = await fetchJobMetrics(payload);
      console.log("📊 Metrics fetched successfully:", JSON.stringify(metrics, null, 2));
    } catch (err) {
      console.error("❌ Failed to fetch metrics:", err?.message || err);
      metrics = "Error fetching metrics";
    }
  }

  try {
    await sendDocketStatus(uuid, status, metrics);
    console.log(`📬 Docket status sent: uuid=${uuid}, status=${status}`);
  } catch (err) {
    console.error("❌ Failed to send Kafka message:", err?.message || err);
  }
  return {
    status: "OK",
    inferenceData: inferenceData,
    buildData: buildData,
    evalData: evalData,
    jobStatus: jobStatus,
  };
}