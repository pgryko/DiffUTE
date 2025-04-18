"""
MinIO utility functions for handling S3-compatible storage operations.

This module provides a clean interface for interacting with MinIO/S3 storage,
replacing the previous pcache_fileio implementation.
"""

import cv2
import numpy as np
from minio import Minio
from typing import Optional, Union, BinaryIO
import io


class MinioHandler:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = True,
    ):
        """
        Initialize MinIO client with credentials and configuration.

        Args:
            endpoint (str): MinIO server endpoint
            access_key (str): Access key for authentication
            secret_key (str): Secret key for authentication
            bucket_name (str): Default bucket to use
            secure (bool): Whether to use HTTPS (default: True)
        """
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket_name = bucket_name

        # Ensure bucket exists
        if not self.client.bucket_exists(bucket_name):
            raise ValueError(f"Bucket {bucket_name} does not exist")

    def download_file(self, object_name: str) -> bytes:
        """
        Download a file from MinIO storage.

        Args:
            object_name (str): Name of the object to download

        Returns:
            bytes: File contents as bytes

        Raises:
            Exception: If download fails
        """
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            return response.read()
        except Exception as e:
            raise Exception(f"Failed to download {object_name}: {str(e)}")
        finally:
            response.close()
            response.release_conn()

    def download_image(self, object_name: str) -> np.ndarray:
        """
        Download and decode an image from MinIO storage.

        Args:
            object_name (str): Name of the image object

        Returns:
            np.ndarray: Decoded image as numpy array

        Raises:
            Exception: If download or decoding fails
        """
        try:
            content = self.download_file(object_name)
            img_array = np.frombuffer(content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            return img
        except Exception as e:
            raise Exception(f"Failed to download/decode image {object_name}: {str(e)}")

    def upload_file(
        self,
        file_data: Union[bytes, BinaryIO],
        object_name: str,
        content_type: Optional[str] = None,
    ) -> None:
        """
        Upload a file to MinIO storage.

        Args:
            file_data: File contents as bytes or file-like object
            object_name (str): Name to give the uploaded object
            content_type (str, optional): Content type of the file

        Raises:
            Exception: If upload fails
        """
        try:
            if isinstance(file_data, bytes):
                file_data = io.BytesIO(file_data)

            file_size = file_data.seek(0, 2)
            file_data.seek(0)

            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=file_data,
                length=file_size,
                content_type=content_type,
            )
        except Exception as e:
            raise Exception(f"Failed to upload {object_name}: {str(e)}")

    def list_objects(self, prefix: str = "", recursive: bool = True):
        """
        List objects in the bucket with optional prefix filtering.

        Args:
            prefix (str): Filter objects by prefix
            recursive (bool): Whether to list objects recursively in directories

        Returns:
            Generator yielding object names
        """
        try:
            objects = self.client.list_objects(
                self.bucket_name, prefix=prefix, recursive=recursive
            )
            return (obj.object_name for obj in objects)
        except Exception as e:
            raise Exception(f"Failed to list objects: {str(e)}")

    def read_json(self, object_name: str) -> dict:
        """
        Read and parse a JSON file from MinIO storage.

        Args:
            object_name (str): Name of the JSON object

        Returns:
            dict: Parsed JSON content

        Raises:
            Exception: If reading or parsing fails
        """
        import json

        try:
            content = self.download_file(object_name)
            return json.loads(content.decode("utf-8"))
        except Exception as e:
            raise Exception(f"Failed to read/parse JSON {object_name}: {str(e)}")
