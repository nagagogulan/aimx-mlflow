import { proxyActivities } from "@temporalio/workflow";

const { helloWorld } = proxyActivities({
  startToCloseTimeout: "5 minute",
  retry: {
    initialInterval: "60 second",
    maximumAttempts: 5,
    backoffCoefficient: 2,
  },
});

export async function runEval(payload) {
  const data = await helloWorld(payload);
  return {
    status: "OK",
    data: data,
  };
}
