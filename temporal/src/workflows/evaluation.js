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
  const uuid = payload.uuid || "unknown-uuid";

  try {
    inferenceData = await runStep("Copying inference scripts", () =>
      copyInferenceScripts(payload)
    );

    buildData = await runStep("Building Docker image", () =>
      buildDockerImage(inferenceData)
    );

    evalData = await runStep("Launching evaluations in cluster", () =>
      runEvaluationsInCluster(payload, inferenceData)
    );

    if (!evalData?.jobName || !evalData?.namespace) {
      throw new Error(`[runEval] Invalid evalData: ${JSON.stringify(evalData)}`);
    }

    jobStatus = await runStep("Waiting for job completion", () =>
      waitForJobCompletion(evalData.jobName, evalData.namespace)
    );

    // ✅ Send success status
    try {
      logStep("Sending docket success status...");
      await sendDocketStatus(uuid, "success");
      logStep("✅ Docket status sent");
    } catch (err) {
      logError("Sending docket status", err);
    }

    logStep("🏁 Workflow completed successfully");

    return {
      status: "OK",
      inferenceData,
      buildData,
      evalData,
      jobStatus,
    };

  } catch (err) {
    // ❌ If any step failed, report failure
    logError("Workflow execution", err);

    try {
      logStep("Sending docket failure status...");
      await sendDocketStatus(uuid, "failed");
      logStep("✅ Failure status sent");
    } catch (sendErr) {
      logError("Sending failure status", sendErr);
    }

    // Ensure the workflow fails in Temporal
    throw new Error(`Workflow failed: ${err.message}`);
  }
}
