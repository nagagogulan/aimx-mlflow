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
  startToCloseTimeout: "20 minute",
  retry: {
    initialInterval: "60 second",
    maximumAttempts: 1,
  },
});

// Utility logger
const logStep = (msg, obj = null) => {
  console.log(`🟢 [runEval] ${msg}`);
  if (obj) console.log(JSON.stringify(obj, null, 2));
};

const logError = (step, err) => {
  console.error(`❌ [runEval] ${step} failed:\n  ↳ ${err.message}`);
};

// Step runner with context
const runStep = async (stepName, fn) => {
  try {
    logStep(`Step: ${stepName}...`);
    const result = await fn();
    logStep(`✅ Step "${stepName}" completed successfully`);
    return result;
  } catch (err) {
    logError(stepName, err);
    throw new Error(`Step "${stepName}" failed: ${err.message}`);
  }
};

export async function runEval(payload) {
  logStep("Step 1: Received payload", payload);

  let inferenceData, buildData, evalData, jobStatus;

  inferenceData = await runStep("Copying inference scripts", () =>
    copyInferenceScripts(payload)
  );

  buildData = await runStep("Building Docker image", () =>
    buildDockerImage(inferenceData)
  );
  console.log("buildData is succeededdddd*************************", buildData)

  evalData = await runStep("Launching evaluations in cluster", () =>
    runEvaluationsInCluster(payload, inferenceData)
  );

    if (!evalData?.jobName || !evalData?.namespace) {
  throw new Error(`[runEval] Invalid evalData: ${JSON.stringify(evalData)}`);
}
  jobStatus = await runStep("Waiting for job completion", () =>
    waitForJobCompletion(evalData.jobName, evalData.namespace)
  );

      console.log("jobStatus is succeededdddd*************************: outputtt", jobStatus)


  if (jobStatus) {
    const uuid = payload.uuid || "unknown-uuid";
    const status = "success";

    try {
      logStep("Sending docket status...");
      await sendDocketStatus(uuid, status);
      logStep("✅ Docket status sent");
    } catch (err) {
      logError("Sending docket status", err);
      // Do not rethrow — this failure won't affect workflow result
    }
  }

  logStep("🏁 Workflow completed successfully");

  return {
    status: "OK",
    inferenceData,
    buildData,
    evalData,
    jobStatus,
  };
}
