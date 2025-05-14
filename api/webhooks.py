"""
Webhook utilities for the Vehicle Registration Card OCR system.
Provides functionality to notify external systems about job status and results.
"""

import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from utils.config import Config

logger = logging.getLogger(__name__)

class WebhookNotifier:
    """Handler for sending webhook notifications."""
    
    def __init__(
        self, 
        webhook_urls: Optional[List[str]] = None,
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        """
        Initialize the webhook notifier.
        
        Args:
            webhook_urls: List of webhook URLs to notify
            timeout: Timeout for webhook requests in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.webhook_urls = webhook_urls or []
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Load webhook URLs from config if not provided
        if not self.webhook_urls:
            config_urls = Config.get("WEBHOOK_URLS", "")
            if config_urls:
                if isinstance(config_urls, str):
                    self.webhook_urls = [url.strip() for url in config_urls.split(",")]
                elif isinstance(config_urls, list):
                    self.webhook_urls = config_urls
    
    async def notify_job_status(
        self, 
        job_id: str, 
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a job status notification to all registered webhooks.
        
        Args:
            job_id: The job ID
            status: Current job status
            metadata: Additional metadata to include
            
        Returns:
            Dictionary with notification results
        """
        if not self.webhook_urls:
            logger.info("No webhook URLs configured, skipping notification")
            return {"success": True, "message": "No webhooks configured"}
        
        # Prepare the payload
        payload = {
            "event": "job_status",
            "job_id": job_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            payload["metadata"] = metadata
            
        # Send notifications to all webhooks
        results = await self._send_to_all_webhooks(payload)
        
        return {
            "success": all(result.get("success", False) for result in results),
            "results": results
        }
    
    async def notify_job_completed(
        self, 
        job_id: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a job completion notification with results to all registered webhooks.
        
        Args:
            job_id: The job ID
            result: The job result data
            
        Returns:
            Dictionary with notification results
        """
        if not self.webhook_urls:
            logger.info("No webhook URLs configured, skipping notification")
            return {"success": True, "message": "No webhooks configured"}
        
        # Prepare the payload
        payload = {
            "event": "job_completed",
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
            
        # Send notifications to all webhooks
        results = await self._send_to_all_webhooks(payload)
        
        return {
            "success": all(result.get("success", False) for result in results),
            "results": results
        }
    
    async def notify_job_failed(
        self, 
        job_id: str,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a job failure notification to all registered webhooks.
        
        Args:
            job_id: The job ID
            error: Error message
            metadata: Additional metadata to include
            
        Returns:
            Dictionary with notification results
        """
        if not self.webhook_urls:
            logger.info("No webhook URLs configured, skipping notification")
            return {"success": True, "message": "No webhooks configured"}
        
        # Prepare the payload
        payload = {
            "event": "job_failed",
            "job_id": job_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            payload["metadata"] = metadata
            
        # Send notifications to all webhooks
        results = await self._send_to_all_webhooks(payload)
        
        return {
            "success": all(result.get("success", False) for result in results),
            "results": results
        }
    
    async def _send_to_all_webhooks(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Send a payload to all registered webhook URLs.
        
        Args:
            payload: The data payload to send
            
        Returns:
            List of results for each webhook
        """
        tasks = []
        for url in self.webhook_urls:
            tasks.append(self._send_webhook(url, payload))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "url": self.webhook_urls[i],
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
                
        return processed_results
    
    async def _send_webhook(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a webhook notification with retries.
        
        Args:
            url: The webhook URL
            payload: The data payload to send
            
        Returns:
            Result of the webhook notification
        """
        result = {
            "url": url,
            "success": False,
            "status_code": None,
            "attempts": 0,
            "error": None
        }
        
        # Try sending webhook with retries
        for attempt in range(1, self.max_retries + 1):
            result["attempts"] = attempt
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        result["status_code"] = response.status
                        
                        if 200 <= response.status < 300:
                            result["success"] = True
                            result["error"] = None
                            return result
                        else:
                            result["error"] = f"HTTP error {response.status}"
                            response_text = await response.text()
                            logger.warning(
                                f"Webhook to {url} failed (attempt {attempt}/{self.max_retries}): "
                                f"HTTP {response.status} - {response_text[:100]}"
                            )
            except asyncio.TimeoutError:
                result["error"] = f"Timeout after {self.timeout} seconds"
                logger.warning(
                    f"Webhook to {url} timed out (attempt {attempt}/{self.max_retries})"
                )
            except Exception as e:
                result["error"] = str(e)
                logger.warning(
                    f"Webhook to {url} failed (attempt {attempt}/{self.max_retries}): {e}"
                )
            
            # If not the last attempt, wait before retrying
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay)
                
        return result
