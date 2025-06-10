import path from "path";
import { NativeConnection, Worker } from "@temporalio/worker";
import * as activities from "../activities/evaluation.js";

async function run() {
  let connection;
  let worker;
  const workflowsPath = path.join("/app", "src", "workflows", "evaluation.js");

  try {
    console.log("🔌 [init] Connecting to Temporal server at 54.251.96.179:7233...");

    connection = await NativeConnection.connect({
      address: "54.251.96.179:7233",
      // TLS/gRPC metadata config could go here
    });

    console.log("✅ [init] Connection to Temporal server established.");

    console.log("🛠️ [worker] Creating worker for task queue 'evaluation'...");
    worker = await Worker.create({
      connection,
      namespace: "default",
      taskQueue: "evaluation",
      workflowsPath,
      activities,
    });

    console.log("🚀 [worker] Worker created. Starting task polling...");
    await worker.run(); // This will block until shutdown or error

    console.log("🛑 [worker] Worker has stopped gracefully.");
  } catch (error) {
    console.error("❌ [error] Worker failed to start or crashed unexpectedly.");
    console.error(`   ↳ Reason: ${error.message}`);
    console.error(`   ↳ Stack: ${error.stack}`);
    process.exit(1);
  }
}

// Entry point
run();
