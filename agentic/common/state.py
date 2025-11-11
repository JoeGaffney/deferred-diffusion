from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Optional

from common.logger import logger
from common.schemas import Media


@dataclass
class Deps:
    images: list[Media] = field(default_factory=list)
    videos: list[Media] = field(default_factory=list)

    def add_or_update_media(self, content_dict: dict, tool_name: str):
        """Add or update media from tool call result"""
        if not isinstance(content_dict, dict):
            return

        task_id = content_dict.get("id")
        if not task_id:
            return

        # Simple and reliable tool name detection
        if tool_name == "images_get":
            target_list = self.images
            media_type = "image"
        elif tool_name == "videos_get":
            target_list = self.videos
            media_type = "video"
        else:
            return

        # Check if media already exists
        existing_media = next((m for m in target_list if m.id == task_id), None)

        if existing_media:
            # Update existing media
            existing_media.status = content_dict.get("status", existing_media.status)
            if "result" in content_dict and isinstance(content_dict["result"], dict):
                result_data = content_dict["result"]
                existing_media.local_file_path = result_data.get("local_file_path", existing_media.local_file_path)
                existing_media.base64_data = result_data.get("base64_data", existing_media.base64_data)
            logger.info(f"Updated {media_type} {task_id} status: {existing_media.status}")
        else:
            # Create new media entry
            result_data = content_dict.get("result", {}) if isinstance(content_dict.get("result"), dict) else {}

            media = Media(
                id=task_id,
                status=content_dict.get("status", "UNKNOWN"),
                model=content_dict.get("model", ""),
                prompt=content_dict.get("prompt", ""),
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

    def get_pending_tasks(self) -> list[Media]:
        return [m for m in self.images + self.videos if m.status in ["PENDING", "STARTED"]]

    def get_pending_task_ids(self) -> list[str]:
        return [m.id for m in self.get_pending_tasks()]

    def get_completed_media(self) -> list[Media]:
        return [m for m in self.images + self.videos if m.status == "SUCCESS"]
