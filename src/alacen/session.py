import atexit
import os
import shutil
from pathlib import Path
from typing import Dict, List


class SessionManager:
    def __init__(self):
        self._counter = 0
        self._sessions: Dict[int, List[Path]] = {}

    def create_session(self) -> int:
        if self._counter in self._sessions:
            raise ValueError("Session already exists with the same ID.")
        session_id = self._counter
        self._sessions[session_id] = []
        self._counter += 1
        return session_id

    def add_resource(self, session_id: int, resource: Path):
        self._sessions[session_id].append(resource)

    def add_resources(self, session_id: int, resources: List[Path]):
        self._sessions[session_id].extend(resources)

    def remove_session(self, session_id: int):
        for resource in self._sessions.pop(session_id):
            if resource.is_dir():
                shutil.rmtree(resource)
            else:
                os.remove(resource)

    def remove_all_sessions(self):
        for session_id in self._sessions.keys():
            self.remove_session(session_id)


manager = SessionManager()


@atexit.register
def clean_up_all_sessions():
    manager.remove_all_sessions()
