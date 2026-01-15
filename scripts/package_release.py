import os
import re
import shutil
import sys
import tarfile


def package_release(version, project_name):
    release_dir = f"releases/{project_name}"

    # 1. Clean and create release directory
    if os.path.exists(release_dir):
        print(f"Cleaning existing release directory: {release_dir}")
        shutil.rmtree(release_dir)
    os.makedirs(release_dir, exist_ok=True)

    # 2. Copy core files
    files_to_copy = ["docker-compose.yml", "docker-compose.comfy.yml", "README.md", "DEPLOYMENT.md"]
    for f in files_to_copy:
        if os.path.exists(f):
            print(f"Copying {f} to {release_dir}")
            shutil.copy(f, release_dir)

    # 3. Copy clients
    if os.path.exists("clients"):
        print(f"Copying clients to {release_dir}")
        shutil.copytree("clients", os.path.join(release_dir, "clients"))

    # 4. Transform Compose Files
    for filename in ["docker-compose.yml", "docker-compose.comfy.yml"]:
        path = os.path.join(release_dir, filename)
        if not os.path.exists(path):
            continue

        print(f"Transforming {filename} for release...")
        with open(path, "r") as f:
            content = f.read()

        # A. Strip build sections (remove build: block and its indented children)
        # This matches 'build:' at any indentation and then all subsequent lines
        # that are indented deeper than the 'build:' line.
        content = re.sub(r"^([ \t]+)build:.*(?:\n\1[ \t]+.*)*\n?", "", content, flags=re.MULTILINE)

        # B. Pin versions (Swap -latest for the specific version)
        # Handles :api-latest, :worker-latest, :comfy-latest
        content = re.sub(r":(api|worker|comfy)-latest", rf":\1-{version}", content)

        with open(path, "w") as f:
            f.write(content)

    # 5. Create the Tarball
    tar_path = f"releases/{project_name}-{version}.tar.gz"
    print(f"Creating archive: {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(release_dir, arcname=project_name)

    print(f"✅ Release package created: {tar_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python package_release.py <version> <project_name>")
        sys.exit(1)
    package_release(sys.argv[1], sys.argv[2])
