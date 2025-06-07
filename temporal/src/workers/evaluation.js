import { NativeConnection, Worker } from "@temporalio/worker";
import * as activities from "../activities/evaluation.js";

async function run() {
  // Step 1: Establish a connection with Temporal server.
  //
  // Worker code uses `@temporalio/worker.NativeConnection`.
  // (But in your application code it's `@temporalio/client.Connection`.)
  
  const connection = await NativeConnection.connect({
    address:"54.251.96.179:7233",
    // TLS and gRPC metadata configuration goes here.
  });
  const workflowsPath = path.join("/app", "src", "workflows", "evaluation.js");

  // Step 2: Register Workflows and Activities with the Worker.
  const worker = await Worker.create({
    connection,
    namespace: "default",
    taskQueue: "evaluation",
    workflowsPath: workflowsPath,
    activities,
  });

  // Step 3: Start accepting tasks on the `hello-world` queue
  //
  // The worker runs until it encounters an unexpected error or the process receives a shutdown signal registered on
  // the SDK Runtime object.
  //
  // By default, worker logs are written via the Runtime logger to STDERR at INFO level.
  //
  // See https://typescript.temporal.io/api/classes/worker.Runtime#install to customize these defaults.
  
    console.log("Worker for task queue 'evaluation' has been created and is starting...");

  await worker.run();
}

run().catch((err) => {
  console.error("Worker encountered an error:", err);
  process.exit(1);
});
