import { Connection, Client } from "@temporalio/client";
import { nanoid } from "nanoid";

async function run() {
  // Connect to the default Server location
  const connection = await Connection.connect({
    address: process.env?.TEMPORAL_ADDRESS,
  });
  // In production, pass options to configure TLS and other settings:
  // {
  //   address: 'foo.bar.tmprl.cloud',
  //   tls: {}
  // }

  const client = new Client({
    connection,
    // namespace: 'foo.bar', // connects to 'default' namespace if not specified
  });

  const payload = {
    taskType: "text-classification",
    modelFramework: "pytorch",
    modelArchitecture: "distilbert",
    modelWeightUrl:
      "http://localhost:5500/inference/text-classification/distilbert/weights/model.pkl",
  };

  const handle = await client.workflow.start("runEval", {
    taskQueue: "evaluation",
    // type inference works! args: [name: string]
    args: [payload],
    // in practice, use a meaningful business ID, like customerId or transactionId
    workflowId: "workflow-" + nanoid(),
  });
  console.log(`Started workflow ${handle.workflowId}`);

  // console.log(handle);

  // optional: wait for client result
  // console.log(await handle.result()); // Hello, Temporal!
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
