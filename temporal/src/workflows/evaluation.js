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
  startToCloseTimeout: "10 minute",
  retry: {
    initialInterval: "60 second",
    maximumAttempts: 1,
    // maximumAttempts: 5,
    // backoffCoefficient: 2,
  },
});

export async function runEval(payload) {
  console.log("📦 Step 1: Received payload from Go backend:");
  console.log(JSON.stringify(payload, null, 2));

  console.log("📁 Step 2: Copying inference scripts...");
  const inferenceData = await copyInferenceScripts(payload);
  console.log("✅ Inference scripts copied:", inferenceData);

  console.log("🐳 Step 3: Building Docker image...");
  const buildData = await buildDockerImage(inferenceData);
  console.log("✅ Docker image built:", buildData);

  console.log("🚀 Step 4: Running evaluations in cluster...");
  const evalData = await runEvaluationsInCluster(payload, inferenceData);
  console.log("✅ Evaluations launched:", evalData);

  console.log("⏳ Step 5: Waiting for job completion...");
  const jobStatus = await waitForJobCompletion(
    evalData.jobName,
    evalData.namespace
  );
  console.log("✅ Job completed with status:", jobStatus);

  if (jobStatus) {
    console.log("📬 Step 6: Sending docket status...");
    const uuidFromProcessedData = payload.uuid; // Replace with real uuid
    const status = "success"; // or "failed"

    await sendDocketStatus(uuidFromProcessedData, status).catch((err) => {
      console.error("❌ Failed to send Kafka message:", err);
    });
    console.log("✅ Docket status sent");
  }

  console.log("🏁 Workflow completed successfully");

  return {
    status: "OK",
    inferenceData,
    buildData,
    evalData,
    jobStatus,
  };
}
