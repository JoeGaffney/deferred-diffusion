import os

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
from aws_cdk import aws_secretsmanager as secretsmanager
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

Need to switch to Sweden to g6e instances or USA
"""


class InfraStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Configuration
        self.redis_port = 6379
        self.api_port = 5000

        # Load secrets
        self.dockerhub_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "DockerHubSecret", "dockerhub-credentials"
        )
        self.hf_token_secret = secretsmanager.Secret.from_secret_name_v2(self, "HfTokenSecret", "hf-token")

        # Setup infrastructure components
        self.setup_vpc_and_security()
        self.setup_efs()
        self.setup_task_role()
        self.setup_cluster_and_namespace()

        # Create services
        self.create_api_service()
        self.create_redis_service()
        self.create_worker_service()
        # self.create_queue_monitor()

    def setup_vpc_and_security(self):
        # VPC (2 AZs, public subnets only)
        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[ec2.SubnetConfiguration(name="Public", subnet_type=ec2.SubnetType.PUBLIC)],
        )

        # Security Group
        self.sg = ec2.SecurityGroup(
            self, "EcsSG", vpc=self.vpc, description="Service security group", allow_all_outbound=True
        )

        # Allow all traffic between resources in the same security group
        self.sg.connections.allow_internally(ec2.Port.all_traffic(), "Allow all internal traffic")

        # Only allow public access to the API port
        self.sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(self.api_port), "Allow public access to API")
        # Add to your security group setup
        self.sg.add_egress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(2049), "Allow outbound NFS traffic")

    def setup_efs(self):
        # EFS for model caching
        self.file_system = efs.FileSystem(
            self,
            "HfCache",
            vpc=self.vpc,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            lifecycle_policy=efs.LifecyclePolicy.AFTER_7_DAYS,
            vpc_subnets={"subnet_type": ec2.SubnetType.PUBLIC},
        )

        # Mount target security group
        self.file_system.connections.allow_default_port_from(self.sg)

    def setup_task_role(self):
        # Common task execution role
        self.task_role = iam.Role(
            self,
            "TaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
            ],
        )

    def setup_cluster_and_namespace(self):
        # ECS Cluster
        self.cluster = ecs.Cluster(self, "Cluster", vpc=self.vpc)

        # Service discovery namespace
        self.namespace = servicediscovery.PrivateDnsNamespace(
            self,
            "ServiceDiscovery",
            name="deferred-diffusion.local",
            vpc=self.vpc,
            description="Private DNS namespace for service discovery",
        )

    def create_api_service(self):
        # Fargate Task for API
        api_task = ecs.FargateTaskDefinition(
            self, "ApiTaskDef", memory_limit_mib=1024, cpu=512, execution_role=self.task_role
        )
        api_container = api_task.add_container(
            "api",
            image=ecs.ContainerImage.from_registry(
                "joegaffney/deferred-diffusion:api-latest", credentials=self.dockerhub_secret
            ),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="api-container", log_retention=cdk.aws_logs.RetentionDays.ONE_DAY
            ),
            environment={
                "DEF_DIF_API_KEYS": os.getenv(
                    "DEF_DIF_API_KEYS", "dummy-key"
                ),  # TODO: Replace with Secrets Manager before production deployment
                "PYTHONUNBUFFERED": "1",
                "CELERY_BROKER_URL": f"redis://redis.deferred-diffusion.local:{self.redis_port}/0",
                "CELERY_RESULT_BACKEND": f"redis://redis.deferred-diffusion.local:{self.redis_port}/1",
            },
        )
        api_container.add_port_mappings(ecs.PortMapping(container_port=self.api_port))

        self.api_service = ecs.FargateService(
            self,
            "ApiService",
            cluster=self.cluster,
            task_definition=api_task,
            desired_count=1,
            assign_public_ip=True,
            security_groups=[self.sg],
            vpc_subnets={"subnet_type": ec2.SubnetType.PUBLIC},
        )

    def create_redis_service(self):
        # Fargate Task for Redis
        redis_task = ecs.FargateTaskDefinition(
            self, "RedisTaskDef", memory_limit_mib=512, cpu=256, execution_role=self.task_role
        )
        redis_task.add_container(
            "redis",
            image=ecs.ContainerImage.from_registry("redis:latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="redis-container", log_retention=cdk.aws_logs.RetentionDays.ONE_DAY
            ),
            port_mappings=[ecs.PortMapping(container_port=self.redis_port)],
        )

        self.redis_service = ecs.FargateService(
            self,
            "RedisService",
            cluster=self.cluster,
            task_definition=redis_task,
            desired_count=1,
            assign_public_ip=True,
            security_groups=[self.sg],
            vpc_subnets={"subnet_type": ec2.SubnetType.PUBLIC},
            cloud_map_options=ecs.CloudMapOptions(
                name="redis",
                cloud_map_namespace=self.namespace,
                dns_record_type=servicediscovery.DnsRecordType.A,
            ),
        )

    def create_worker_service(self):
        # Create role with proper EFS permissions
        worker_task_role = iam.Role(self, "WorkerTaskRole", assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"))
        worker_task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "elasticfilesystem:ClientMount",
                    "elasticfilesystem:ClientRootAccess",
                    "elasticfilesystem:ClientWrite",
                ],
                resources=[self.file_system.file_system_arn],
            )
        )

        # Create task definition with the role assigned
        worker_task = ecs.Ec2TaskDefinition(
            self,
            "WorkerTaskDef",
            network_mode=ecs.NetworkMode.BRIDGE,
            task_role=worker_task_role,
            execution_role=self.task_role,  # Add execution role as well
        )

        volume_name = "hf_cache"
        worker_task.add_volume(
            name=volume_name,
            efs_volume_configuration=ecs.EfsVolumeConfiguration(
                file_system_id=self.file_system.file_system_id,
                transit_encryption="ENABLED",
                authorization_config=ecs.AuthorizationConfig(iam="ENABLED"),
            ),
        )
        worker_container = worker_task.add_container(
            "worker",
            image=ecs.ContainerImage.from_registry(
                "joegaffney/deferred-diffusion:worker-latest", credentials=self.dockerhub_secret
            ),
            # Add logging configuration here:
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="worker-container", log_retention=cdk.aws_logs.RetentionDays.ONE_DAY
            ),
            # memory_limit_mib=57344,  # ~56GB (out of 64GB total on g5.4xlarge)
            # cpu=15360,  # ~15 vCPUs (out of 16 total on g5.4xlarge)
            memory_reservation_mib=1024 * 12,  # Soft limit (1GB)
            environment={
                "PYTHONUNBUFFERED": "1",
                "OPENAI_API_KEY": "dummy-key",
                "RUNWAYML_API_SECRET": "dummy-secret",
                "HF_HOME": "/WORKSPACE",
                "TORCH_HOME": "/WORKSPACE",
                "CELERY_BROKER_URL": f"redis://redis.deferred-diffusion.local:{self.redis_port}/0",
                "CELERY_RESULT_BACKEND": f"redis://redis.deferred-diffusion.local:{self.redis_port}/1",
            },
            secrets={
                "HF_TOKEN": ecs.Secret.from_secrets_manager(self.hf_token_secret),
            },
            gpu_count=1,
        )

        worker_container.add_mount_points(
            ecs.MountPoint(container_path="/WORKSPACE", source_volume=volume_name, read_only=False)
        )

        # EC2 Capacity
        asg = autoscaling.AutoScalingGroup(
            self,
            "WorkerAutoScalingGroup",
            vpc=self.vpc,
            instance_type=ec2.InstanceType("g5.xlarge"),
            machine_image=ecs.EcsOptimizedImage.amazon_linux2(hardware_type=ecs.AmiHardwareType.GPU),
            desired_capacity=1,
            vpc_subnets={"subnet_type": ec2.SubnetType.PUBLIC},
            associate_public_ip_address=True,
            security_group=self.sg,  # This is critical for EFS access
        )

        # Add the ASG to the cluster
        capacity_provider = ecs.AsgCapacityProvider(self, "AsgCapacityProvider", auto_scaling_group=asg)
        self.cluster.add_asg_capacity_provider(capacity_provider)

        # Create the EC2 Service using the capacity provider
        self.worker_service = ecs.Ec2Service(
            self,
            "WorkerService",
            cluster=self.cluster,
            task_definition=worker_task,
            desired_count=1,
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(capacity_provider=capacity_provider.capacity_provider_name, weight=1)
            ],
        )
        # Spot Instances for cost savings
        # self.cluster.add_capacity(
        #     "DefaultAutoScalingGroup",
        #     instance_type=ec2.InstanceType("g5.4xlarge"),
        #     desired_capacity=1,
        #     vpc_subnets={"subnet_type": ec2.SubnetType.PUBLIC},
        #     spot_price="2.00",  # Maximum price you're willing to pay per hour (in USD)
        #     spot_instance_draining=True,  # Allow for graceful termination
        # )

    def create_queue_monitor(self):
        queue_monitor_role = iam.Role(
            self,
            "QueueMonitorRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
                iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchFullAccess"),
            ],
        )

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

        queue_monitor = lambda_.Function(
            self,
            "QueueMonitorFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="queue_monitor.handler",
            code=lambda_.Code.from_asset("lambda"),
            vpc=self.vpc,
            security_groups=[self.sg],
            timeout=cdk.Duration.seconds(30),
            environment={
                "REDIS_HOST": "redis.deferred-diffusion.local",
                "REDIS_PORT": str(self.redis_port),
                "CLUSTER_NAME": self.cluster.cluster_name,
                "SERVICE_NAME": "WorkerService",
                "MIN_TASKS_THRESHOLD": "1",
                "MAX_QUEUE_AGE_SECONDS": "600",
            },
            role=iam.Role.from_role_arn(self, "ImportedQueueMonitorRole", role_arn=queue_monitor_role.role_arn),
        )

        rule = events.Rule(self, "ScheduleRule", schedule=events.Schedule.rate(cdk.Duration.minutes(1)))
        rule.add_target(targets.LambdaFunction(queue_monitor))
