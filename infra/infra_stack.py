import aws_cdk as cdk
from aws_cdk import Stack
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_efs as efs
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_secretsmanager as secretsmanager  # Add this import
from aws_cdk import aws_servicediscovery as servicediscovery
from constructs import Construct

"""
# GPU cost notes
Instance Size	GPU (GB)  	vCPUs	Memory(GiB)	Storage (GB)  	EUR  				USA
g6e.xlarge		48			4	    32		    250		        2.327 USD 	        1.861 USD per Hour
g6e.2xlarge		48			8	    64		    450		        2.8035 USD 	        2.24208 USD per Hour
g5.4xlarge		24			16	    64		    1x600						        $1.624	


Ireland viable
g5.2xlarge
g5.4xlarge

Need to swich to Sweaden to g6e instances or USA
"""


class InfraStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # PORTS
        redis_port = 6379

        # Reference the existing secret
        dockerhub_secret = secretsmanager.Secret.from_secret_name_v2(self, "DockerHubSecret", "dockerhub-credentials")
        hf_token_secret = secretsmanager.Secret.from_secret_name_v2(self, "HfTokenSecret", "hf-token")

        # VPC (2 AZs, public subnets only)
        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[ec2.SubnetConfiguration(name="Public", subnet_type=ec2.SubnetType.PUBLIC)],
        )

        # Security Group (allows internal communication, restricts external access)
        sg = ec2.SecurityGroup(self, "EcsSG", vpc=vpc, description="Service security group", allow_all_outbound=True)

        # Allow all traffic between resources in the same security group
        sg.connections.allow_internally(ec2.Port.all_traffic(), "Allow all internal traffic")

        # Only allow public access to the API port
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(5000), "Allow public access to API")

        # ECS Cluster
        cluster = ecs.Cluster(self, "Cluster", vpc=vpc)

        # Add this after creating the cluster
        namespace = servicediscovery.PrivateDnsNamespace(
            self,
            "ServiceDiscovery",
            name="deferred-diffusion.local",
            vpc=vpc,
            description="Private DNS namespace for service discovery",
        )

        # EFS
        file_system = efs.FileSystem(
            self,
            "HfCache",
            vpc=vpc,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            lifecycle_policy=efs.LifecyclePolicy.AFTER_7_DAYS,
        )

        # Mount target security group (optional, uses same as ECS for simplicity)
        file_system.connections.allow_default_port_from(sg)

        # Task Role
        task_role = iam.Role(
            self,
            "TaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
            ],
        )

        # Fargate Task for API
        api_task = ecs.FargateTaskDefinition(
            self, "ApiTaskDef", memory_limit_mib=1024, cpu=512, execution_role=task_role
        )
        api_container = api_task.add_container(
            "api",
            image=ecs.ContainerImage.from_registry(
                "joegaffney/deferred-diffusion:api-latest", credentials=dockerhub_secret
            ),
            environment={
                "PYTHONUNBUFFERED": "1",
                "HF_HOME": "/WORKSPACE",
                "TORCH_HOME": "/WORKSPACE",
                "CELERY_BROKER_URL": f"redis://redis.deferred-diffusion.local:{redis_port}/0",
                "CELERY_RESULT_BACKEND": f"redis://redis.deferred-diffusion.local:{redis_port}/1",
            },
        )
        api_container.add_port_mappings(ecs.PortMapping(container_port=5000))

        api_service = ecs.FargateService(
            self,
            "ApiService",
            cluster=cluster,
            task_definition=api_task,
            desired_count=1,
            assign_public_ip=True,
            security_groups=[sg],
            vpc_subnets={"subnet_type": ec2.SubnetType.PUBLIC},
        )

        # Fargate Task for Redis
        redis_task = ecs.FargateTaskDefinition(
            self, "RedisTaskDef", memory_limit_mib=512, cpu=256, execution_role=task_role
        )
        redis_task.add_container(
            "redis",
            image=ecs.ContainerImage.from_registry("redis:latest"),
            port_mappings=[ecs.PortMapping(container_port=redis_port)],
        )

        redis_service = ecs.FargateService(
            self,
            "RedisService",
            cluster=cluster,
            task_definition=redis_task,
            desired_count=1,
            assign_public_ip=True,
            security_groups=[sg],
            vpc_subnets={"subnet_type": ec2.SubnetType.PUBLIC},
            cloud_map_options=ecs.CloudMapOptions(
                name="redis",  # This creates redis.deferred-diffusion.local
                cloud_map_namespace=namespace,
                dns_record_type=servicediscovery.DnsRecordType.A,
            ),
        )

        def worker_task():
            worker_task = ecs.Ec2TaskDefinition(self, "WorkerTaskDef", network_mode=ecs.NetworkMode.AWS_VPC)
            volume_name = "hf_cache"
            worker_task.add_volume(
                name=volume_name,
                efs_volume_configuration=ecs.EfsVolumeConfiguration(file_system_id=file_system.file_system_id),
            )

            worker_container = worker_task.add_container(
                "worker",
                image=ecs.ContainerImage.from_registry(
                    "joegaffney/deferred-diffusion:worker-latest", credentials=dockerhub_secret
                ),
                memory_limit_mib=57344,  # ~56GB (out of 64GB total on g5.4xlarge)
                cpu=15360,  # ~15 vCPUs (out of 16 total on g5.4xlarge)
                environment={
                    "PYTHONUNBUFFERED": "1",
                    "OPENAI_API_KEY": "dummy-key",
                    "RUNWAYML_API_SECRET": "dummy-secret",
                    "HF_HOME": "/WORKSPACE",
                    "TORCH_HOME": "/WORKSPACE",
                    "CELERY_BROKER_URL": f"redis://redis.deferred-diffusion.local:{redis_port}/0",
                    "CELERY_RESULT_BACKEND": f"redis://redis.deferred-diffusion.local:{redis_port}/1",
                },
                secrets={
                    "HF_TOKEN": ecs.Secret.from_secrets_manager(hf_token_secret),
                },
            )

            # Add mount points to the container after creation
            worker_container.add_mount_points(
                ecs.MountPoint(container_path="/WORKSPACE", source_volume=volume_name, read_only=False)
            )

            # EC2 Capacity
            cluster.add_capacity(
                "DefaultAutoScalingGroup",
                instance_type=ec2.InstanceType("g5.4xlarge"),
                desired_capacity=1,
                vpc_subnets={"subnet_type": ec2.SubnetType.PUBLIC},
            )

            ecs.Ec2Service(
                self,
                "WorkerService",
                cluster=cluster,
                task_definition=worker_task,
                desired_count=1,
                # assign_public_ip=True,
                security_groups=[sg],
                vpc_subnets={"subnet_type": ec2.SubnetType.PUBLIC},
            )

        worker_task()

        def queue_monitor_task():
            # Create the Lambda function from a file instead of inline code
            queue_monitor_role = iam.Role(
                self,
                "QueueMonitorRole",
                assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
                    iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchFullAccess"),
                ],
            )

            # Add ECS and EC2 permissions
            queue_monitor_role.add_to_policy(
                iam.PolicyStatement(
                    actions=[
                        "ecs:UpdateService",
                        "ecs:DescribeServices",
                        "ec2:CreateNetworkInterface",
                        "ec2:DescribeNetworkInterfaces",
                        "ec2:DeleteNetworkInterface",
                    ],
                    resources=["*"],
                )
            )

            # Create Lambda from a separate file
            queue_monitor = lambda_.Function(
                self,
                "QueueMonitorFunction",
                runtime=lambda_.Runtime.PYTHON_3_9,
                handler="queue_monitor.handler",
                code=lambda_.Code.from_asset("lambda"),  # Put your Lambda code in a 'lambda' directory
                # code=lambda_.Code.from_asset("../lambda"),
                vpc=vpc,
                security_groups=[sg],
                timeout=cdk.Duration.seconds(30),
                environment={
                    "REDIS_HOST": "redis.deferred-diffusion.local",
                    "REDIS_PORT": str(redis_port),
                    "CLUSTER_NAME": cluster.cluster_name,
                    "SERVICE_NAME": "WorkerService",
                    "MIN_TASKS_THRESHOLD": "1",  # Minimum task threshold to scale up from 0
                    "MAX_QUEUE_AGE_SECONDS": "600",  # How long a task can sit in queue before scaling
                },
                # Pass the role as ARN instead of directly
                role=iam.Role.from_role_arn(self, "ImportedQueueMonitorRole", role_arn=queue_monitor_role.role_arn),
                # role=queue_monitor_role,
            )

            # Schedule Lambda to run every minute
            rule = events.Rule(self, "ScheduleRule", schedule=events.Schedule.rate(cdk.Duration.minutes(1)))
            rule.add_target(targets.LambdaFunction(queue_monitor))  # type: ignore
