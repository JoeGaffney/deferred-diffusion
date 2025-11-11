from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Optional

from common.logger import logger
from common.schemas import Media


@dataclass
class Deps:
    images: list[Media] = field(default_factory=list)
    videos: list[Media] = field(default_factory=list)
    version: int = 0  # increments on any mutation

    def clone(self) -> "Deps":
        # New instance, new lists (same Media objects)
        return Deps(images=list(self.images), videos=list(self.videos))

    def add_or_update_media(self, call_dict: dict, tool_name: str):
        """Add or update media from tool call result"""

        if not isinstance(call_dict, dict):
            return

        task_id = call_dict.get("id")
        if not task_id:
            return

        logger.debug(
            f"Processing media update for tool {tool_name}, task ID {task_id}, content keys: {list(call_dict.keys())}"
        )

        # Simple and reliable tool name detection
        if tool_name in ["images_get", "images_create"]:
            target_list = self.images
            media_type = "image"
        elif tool_name in ["videos_get", "videos_create"]:
            target_list = self.videos
            media_type = "video"
        else:
            return

        # Check if media already exists
        existing_media = next((m for m in target_list if m.id == task_id), None)

        if existing_media:
            # Update existing media
            existing_media.status = call_dict.get("status", existing_media.status)
            if "result" in call_dict and isinstance(call_dict["result"], dict):
                result_data = call_dict["result"]
                existing_media.local_file_path = result_data.get("local_file_path", existing_media.local_file_path)
                existing_media.base64_data = result_data.get("base64_data", existing_media.base64_data)
            logger.warning(f"Updated {media_type} {task_id} status: {existing_media.status}")
        else:
            # Create new media entry
            result_data = call_dict.get("result", {}) if isinstance(call_dict.get("result"), dict) else {}

            media = Media(
                id=task_id,
                status=call_dict.get("status", "UNKNOWN"),
                local_file_path=result_data.get("local_file_path", ""),
                base64_data=result_data.get("base64_data", ""),
                type=media_type,
            )

            target_list.append(media)
            logger.info(f"Added new {media_type} {task_id} with status: {media.status}")

    def get_image_by_id(self, image_id: str) -> Optional[Media]:
        return next((img for img in self.images if img.id == image_id), None)

    def get_video_by_id(self, video_id: str) -> Optional[Media]:
        return next((vid for vid in self.videos if vid.id == video_id), None)

    def get_pending_images_ids(self) -> list[str]:
        return [m.id for m in self.images if m.status != "SUCCESS"]

    def get_pending_videos_ids(self) -> list[str]:
        return [m.id for m in self.videos if m.status != "SUCCESS"]

    def get_completed_media(self) -> list[Media]:
        return [m for m in self.images + self.videos if m.status == "SUCCESS"]
