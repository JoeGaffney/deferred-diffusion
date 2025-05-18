import json
import logging
import os
import time
from datetime import datetime

import boto3
import redis

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_queue_metrics(redis_client):
    """Get comprehensive metrics about the Celery queue"""
    metrics = {"queue_length": 0, "oldest_task_age": 0, "has_high_priority": False}

    try:
        # Get queue length
        queue_length = redis_client.llen("celery")
        metrics["queue_length"] = queue_length

        # Check if there are any tasks
        if queue_length > 0:
            # Try to get the timestamp of the oldest task
            # Note: This depends on your Celery configuration and how tasks are stored
            # This is a simplified example and may need adaptation to your specific setup
            all_tasks = redis_client.lrange("celery", 0, -1)

            if all_tasks:
                # Check for any high priority tasks (you would need to define what makes a task high priority)
                # This is just a placeholder example
                for task in all_tasks:
                    try:
                        task_data = json.loads(task)
                        if task_data.get("priority", 0) > 8:  # Example priority threshold
                            metrics["has_high_priority"] = True
                            break
                    except:
                        pass

                # Try to get the oldest task timestamp
                try:
                    oldest_task = redis_client.lindex("celery", -1)  # Last item is the oldest in FIFO queue
                    task_data = json.loads(oldest_task)
                    if "eta" in task_data:
                        task_time = datetime.fromisoformat(task_data["eta"].replace("Z", "+00:00"))
                        metrics["oldest_task_age"] = (datetime.now() - task_time).total_seconds()
                except:
                    logger.warning("Could not parse task timestamp")

    except Exception as e:
        logger.error(f"Error getting queue metrics: {e}")

    return metrics


def should_scale_up(metrics, current_count):
    """Determine if we should scale up based on queue metrics"""
    min_tasks = int(os.environ.get("MIN_TASKS_THRESHOLD", "1"))
    max_age = int(os.environ.get("MAX_QUEUE_AGE_SECONDS", "600"))  # 10 minutes default

    # Scale up from 0 if we have tasks
    if current_count == 0 and metrics["queue_length"] >= min_tasks:
        logger.info(f"Scaling up from 0 because queue has {metrics['queue_length']} tasks")
        return True

    # Scale up if we have high priority tasks
    if metrics["has_high_priority"]:
        logger.info("Scaling up because high priority tasks detected")
        return True

    # Scale up if tasks are waiting too long
    if metrics["oldest_task_age"] > max_age and metrics["queue_length"] > current_count:
        logger.info(f"Scaling up because oldest task is {metrics['oldest_task_age']} seconds old")
        return True

    # Scale up if queue is significantly larger than worker count
    if metrics["queue_length"] > current_count * 3:  # Each worker can handle ~3 tasks
        logger.info(f"Scaling up because queue length ({metrics['queue_length']}) exceeds worker capacity")
        return True

    return False


def should_scale_down(metrics, current_count):
    """Determine if we should scale down based on queue metrics"""
    # Don't scale down if there are tasks in the queue
    if metrics["queue_length"] > 0:
        return False

    # Scale down to 0 if queue is empty and we have workers
    if metrics["queue_length"] == 0 and current_count > 0:
        logger.info("Scaling down to 0 because queue is empty")
        return True

    return False


def handler(event, context):
    # Get environment variables
    redis_host = os.environ["REDIS_HOST"]
    redis_port = int(os.environ["REDIS_PORT"])
    cluster_name = os.environ["CLUSTER_NAME"]
    service_name = os.environ["SERVICE_NAME"]

    try:
        # Connect to Redis with timeout and error handling
        try:
            redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=0,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            redis_client.ping()  # Test connection
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return {"statusCode": 500, "body": json.dumps("Failed to connect to Redis")}

        # Get the queue metrics
        queue_metrics = get_queue_metrics(redis_client)

        # Publish metrics to CloudWatch
        cloudwatch = boto3.client("cloudwatch")
        cloudwatch.put_metric_data(
            Namespace="CeleryMetrics",
            MetricData=[
                {"MetricName": "QueueLength", "Value": queue_metrics["queue_length"], "Unit": "Count"},
                {"MetricName": "OldestTaskAge", "Value": queue_metrics["oldest_task_age"], "Unit": "Seconds"},
                {
                    "MetricName": "HasHighPriorityTasks",
                    "Value": 1 if queue_metrics["has_high_priority"] else 0,
                    "Unit": "Count",
                },
            ],
        )

        # Get current ECS service state
        ecs = boto3.client("ecs")
        try:
            current_service = ecs.describe_services(cluster=cluster_name, services=[service_name])["services"][0]
            current_count = current_service["desiredCount"]
        except Exception as e:
            logger.error(f"Failed to get ECS service info: {e}")
            return {"statusCode": 500, "body": json.dumps("Failed to get ECS service info")}

        # Determine if we should scale
        should_update = False
        new_count = current_count

        if should_scale_up(queue_metrics, current_count):
            new_count = min(current_count + 1, 5)  # Increase by 1, max of 5
            should_update = new_count != current_count
        elif should_scale_down(queue_metrics, current_count):
            new_count = 0  # Scale to zero if needed
            should_update = True

        # Update service if needed
        if should_update:
            try:
                ecs.update_service(cluster=cluster_name, service=service_name, desiredCount=new_count)
                logger.info(f"Updated service desired count from {current_count} to {new_count}")
            except Exception as e:
                logger.error(f"Failed to update service: {e}")
                return {"statusCode": 500, "body": json.dumps(f"Failed to update service: {str(e)}")}
        else:
            logger.info(
                f"No scaling action needed. Queue length: {queue_metrics['queue_length']}, Current count: {current_count}"
            )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "queue_metrics": queue_metrics,
                    "current_count": current_count,
                    "new_count": new_count,
                    "updated": should_update,
                }
            ),
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"statusCode": 500, "body": json.dumps(f"Error: {str(e)}")}
