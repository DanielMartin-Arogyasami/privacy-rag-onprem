"""
Audit Logger — Constraint C4 (Auditability)
FIX #5: Uses fcntl.flock() for atomic writes across multiple uvicorn workers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from src.models import AuditRecord

logger = logging.getLogger(__name__)

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


class AuditLogger:
    """Append-only JSON audit log with file locking for multi-worker safety."""

    def __init__(self, log_dir: str | Path = "./logs/audit", enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        if enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, record: AuditRecord) -> None:
        if not self.enabled:
            return
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"
        line = record.model_dump_json() + "\n"

        # FIX #5: File locking for multi-worker atomicity (POSIX); plain append on Windows
        with open(log_file, "a", encoding="utf-8") as f:
            if _HAS_FCNTL:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            else:
                f.write(line)
                f.flush()

        logger.debug("Audit record %s written", record.audit_id)

    def query_logs(self, user_id: str | None = None, start: datetime | None = None, end: datetime | None = None) -> list[AuditRecord]:
        records = []
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = AuditRecord.model_validate_json(line)
                    if user_id and rec.user_id != user_id:
                        continue
                    if start and rec.timestamp < start:
                        continue
                    if end and rec.timestamp > end:
                        continue
                    records.append(rec)
        return records
