"""
Scan Job Manager
Manages background ML scan jobs with progress tracking and persistence
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import threading


class ScanJobManager:
    """
    Manages background scan jobs with progress tracking
    Jobs persist across Flask restarts via JSON file
    """

    def __init__(self, state_file: str = 'backend/data/scan_jobs.json'):
        self.state_file = state_file
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._load_state()
        self._cleanup_old_jobs()

    def _load_state(self):
        """Load job state from disk"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.jobs = json.load(f)
                print(f"[JOB MANAGER] Loaded {len(self.jobs)} jobs from state file")
            except Exception as e:
                print(f"[JOB MANAGER] Error loading state: {e}")
                self.jobs = {}
        else:
            self.jobs = {}

    def _save_state(self):
        """Save job state to disk"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.jobs, f, indent=2)
        except Exception as e:
            print(f"[JOB MANAGER] Error saving state: {e}")

    def _cleanup_old_jobs(self):
        """Remove completed jobs older than 1 hour"""
        cutoff_time = datetime.now() - timedelta(hours=1)

        with self.lock:
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if job['status'] in ['completed', 'failed']:
                    job_time = datetime.fromisoformat(job['updated_at'])
                    if job_time < cutoff_time:
                        jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self.jobs[job_id]

            if jobs_to_remove:
                print(f"[JOB MANAGER] Cleaned up {len(jobs_to_remove)} old jobs")
                self._save_state()

    def create_job(self, total_items: int) -> str:
        """
        Create a new scan job

        Args:
            total_items: Total number of items to scan

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())

        with self.lock:
            self.jobs[job_id] = {
                'job_id': job_id,
                'status': 'running',
                'progress': 0,
                'total': total_items,
                'current_symbol': None,
                'percentage': 0.0,
                'started_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'completed_at': None,
                'error': None,
                'signals_count': 0
            }
            self._save_state()

        print(f"[JOB MANAGER] Created job {job_id} for {total_items} items")
        return job_id

    def update_progress(self, job_id: str, current: int, symbol: str = None):
        """
        Update job progress

        Args:
            job_id: Job identifier
            current: Current item number
            symbol: Current symbol being processed
        """
        with self.lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]
            job['progress'] = current
            job['current_symbol'] = symbol
            job['percentage'] = (current / job['total']) * 100 if job['total'] > 0 else 0
            job['updated_at'] = datetime.now().isoformat()

            # Save state every 10 items
            if current % 10 == 0:
                self._save_state()

    def complete_job(self, job_id: str, signals_count: int):
        """
        Mark job as completed

        Args:
            job_id: Job identifier
            signals_count: Number of signals generated
        """
        with self.lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]
            job['status'] = 'completed'
            job['progress'] = job['total']
            job['percentage'] = 100.0
            job['signals_count'] = signals_count
            job['completed_at'] = datetime.now().isoformat()
            job['updated_at'] = datetime.now().isoformat()
            self._save_state()

        print(f"[JOB MANAGER] Job {job_id} completed with {signals_count} signals")

    def fail_job(self, job_id: str, error: str):
        """
        Mark job as failed

        Args:
            job_id: Job identifier
            error: Error message
        """
        with self.lock:
            if job_id not in self.jobs:
                return

            job = self.jobs[job_id]
            job['status'] = 'failed'
            job['error'] = error
            job['completed_at'] = datetime.now().isoformat()
            job['updated_at'] = datetime.now().isoformat()
            self._save_state()

        print(f"[JOB MANAGER] Job {job_id} failed: {error}")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status

        Args:
            job_id: Job identifier

        Returns:
            Job dict or None if not found
        """
        with self.lock:
            return self.jobs.get(job_id)

    def get_active_job(self) -> Optional[Dict[str, Any]]:
        """
        Get currently running job if any

        Returns:
            Active job dict or None
        """
        with self.lock:
            for job in self.jobs.values():
                if job['status'] == 'running':
                    return job
            return None


# Global instance
_job_manager = None

def get_job_manager() -> ScanJobManager:
    """Get global job manager instance"""
    global _job_manager
    if _job_manager is None:
        _job_manager = ScanJobManager()
    return _job_manager
