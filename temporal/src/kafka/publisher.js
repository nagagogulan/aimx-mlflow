import { Kafka } from "kafkajs";
const { createTopicIfNotExists } = require('./worker.js');
 
async function sendDocketStatus(uuid, status) {
  const broker = process.env?.KAFKA_ADDRESS;
  const topic = "docket-status";
 
  // Create topic if it doesn't exist
  await createTopicIfNotExists(broker, topic);
 
  const kafka = new Kafka({ brokers: [broker] });
  const producer = kafka.producer();
  await producer.connect();
 
  const message = {
    uuid,
    status
  };
 
  await producer.send({
    topic,
    messages: [{
      key: uuid,
      value: JSON.stringify(message)
    }]
  });
 
  console.log("Message sent to topic 'docket-status':", message);
 
  await producer.disconnect();
  
}

module.exports = {
  sendDocketStatus
};