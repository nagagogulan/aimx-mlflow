import * as k8s from "@kubernetes/client-node";
async function main() {
  const kc = new k8s.KubeConfig();
  kc.loadFromDefault(); // This will load from ~/.kube/config
  const k8sApi = kc.makeApiClient(k8s.CoreV1Api);
  const k8sBatchApi = kc.makeApiClient(k8s.BatchV1Api);
  const start = Date.now();

  const jobName = "aimx-evaluation-kvax";
  const namespace = "default";
  const timeoutMs = 600000;
  const pollInterval = 5000;

  while (true) {
    const job = await k8sBatchApi.readNamespacedJob({
      name: jobName,
      namespace,
    });
    // const job = res.body;

    // console.log(JSON.stringify(res, null, 2));

    if (job.status.succeeded === 1) {
      console.log(`✅ Job ${jobName} completed successfully.`);
      return true;
    }

    if (job.status.failed && job.status.failed > 0) {
      throw new Error(
        `❌ Job ${jobName} failed with ${job.status.failed} failures.`
      );
    }

    if (Date.now() - start > timeoutMs) {
      throw new Error(`⏰ Timeout waiting for Job ${jobName} to complete.`);
    }

    await new Promise((resolve) => setTimeout(resolve, pollInterval));
  }
}

main();
