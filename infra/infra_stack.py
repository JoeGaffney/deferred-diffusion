import aws_cdk as cdk
from aws_cdk import Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_efs as efs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_secretsmanager as secretsmanager  # Add this import
from constructs import Construct


class InfraStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Reference the existing secret
        dockerhub_secret = secretsmanager.Secret.from_secret_name_v2(self, "DockerHubSecret", "dockerhub-credentials")

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
                "CELERY_BROKER_URL": "redis://redis:6379/0",
                "CELERY_RESULT_BACKEND": "redis://redis:6379/1",
            },
        )
        api_container.add_port_mappings(ecs.PortMapping(container_port=5000))

        ecs.FargateService(
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
            port_mappings=[ecs.PortMapping(container_port=6379)],
        )

        ecs.FargateService(
            self,
            "RedisService",
            cluster=cluster,
            task_definition=redis_task,
            desired_count=1,
            assign_public_ip=True,
            security_groups=[sg],
            vpc_subnets={"subnet_type": ec2.SubnetType.PUBLIC},
        )

        enable_workers = False
        if enable_workers:
            # EC2 Task for Worker (to support GPU if needed later)
            worker_task = ecs.Ec2TaskDefinition(self, "WorkerTaskDef", network_mode=ecs.NetworkMode.AWS_VPC)
            volume_name = "hf_cache"
            worker_task.add_volume(
                name=volume_name,
                efs_volume_configuration=ecs.EfsVolumeConfiguration(file_system_id=file_system.file_system_id),
            )

            worker_container = worker_task.add_container(
                "worker",
                image=ecs.ContainerImage.from_registry("joegaffney/deferred-diffusion:worker-latest"),
                memory_limit_mib=2048,
                cpu=1024,
                environment={
                    "PYTHONUNBUFFERED": "1",
                    "OPENAI_API_KEY": "dummy-key",
                    "RUNWAYML_API_SECRET": "dummy-secret",
                    "HF_TOKEN": "dummy-token",
                    "HF_HOME": "/WORKSPACE",
                    "TORCH_HOME": "/WORKSPACE",
                    "CELERY_BROKER_URL": "redis://redis:6379/0",
                    "CELERY_RESULT_BACKEND": "redis://redis:6379/1",
                },
                # mount_points=[ecs.MountPoint(container_path="/WORKSPACE", source_volume=volume_name, read_only=False)],
            )

            # Add mount points to the container after creation
            worker_container.add_mount_points(
                ecs.MountPoint(container_path="/WORKSPACE", source_volume=volume_name, read_only=False)
            )

            # EC2 Capacity
            cluster.add_capacity(
                "DefaultAutoScalingGroup",
                instance_type=ec2.InstanceType("t3.large"),
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
