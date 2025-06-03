// kafka.js
 
import { Kafka, logLevel } from "kafkajs";

 
/**
* Create a Kafka topic if it does not exist
*/
export async function createTopicIfNotExists(broker, topic, numPartitions = 1, replicationFactor = 1) {
  const kafka = new Kafka({ brokers: [broker], logLevel: logLevel.NOTHING });
  const admin = kafka.admin();
  await admin.connect();
 
  const topics = await admin.listTopics();
 
  if (!topics.includes(topic)) {
    await admin.createTopics({
      topics: [{
        topic,
        numPartitions,
        replicationFactor,
      }],
    });
    console.log(`Successfully created topic: ${topic}`);
  } else {
    console.log(`Topic ${topic} already exists`);
  }
 
  await admin.disconnect();
}
 
/**
* Produce a message with filePath and chunkTopic to filePathTopic
*/
export async function produceDocketFilePath(filePath, filePathTopic, chunkTopic, broker) {
  await createTopicIfNotExists(broker, filePathTopic);
 
  const kafka = new Kafka({ brokers: [broker] });
  const producer = kafka.producer();
 
  await producer.connect();
 
  const message = {
    file_path: filePath,
    chunk_topic: chunkTopic
  };
 
  await producer.send({
    topic: filePathTopic,
    messages: [{ value: JSON.stringify(message) }]
  });
 
  await producer.disconnect();
}
 
/**
* Get Kafka consumer for a topic and group
*/
export function getKafkaReader(topic, groupId, broker) {
  const kafka = new Kafka({ brokers: [broker] });
 
  return kafka.consumer({ groupId });
}
 
/**
* Get Kafka producer for a topic
*/
export async function getKafkaWriter(topic, broker) {
  await createTopicIfNotExists(broker, topic);
 
  const kafka = new Kafka({ brokers: [broker] });
  const producer = kafka.producer();
  await producer.connect();
  return producer;
}
 
/**
* Publish an audit log message to a topic
*/
export async function publishAuditLog(auditLog, broker, topic) {
  const producer = await getKafkaWriter(topic, broker);
 
  await producer.send({
    topic,
    messages: [{ value: JSON.stringify(auditLog) }]
  });
 
  console.log("Audit log published:", auditLog);
 
  await producer.disconnect();
}

