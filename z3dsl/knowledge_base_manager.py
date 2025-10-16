import json
import logging
from typing import Any, Dict, List, Optional
import difflib

logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    def __init__(self, kb_file_path: str = "knowledge_base.json"):
        self.kb_file_path = kb_file_path
        self.knowledge_base: Dict[str, Any] = self._load_knowledge_base()

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Loads the knowledge base from a JSON file."""
        try:
            with open(self.kb_file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Knowledge base file not found at {self.kb_file_path}. Starting with an empty knowledge base.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding knowledge base JSON from {self.kb_file_path}: {e}")
            return {}

    def _save_knowledge_base(self) -> None:
        """Saves the current knowledge base to a JSON file."""
        with open(self.kb_file_path, "w") as f:
            json.dump(self.knowledge_base, f, indent=2)

    def propose_additions(
        self,
        kb_additions: Optional[List[str]] = None,
        func_additions: Optional[List[Dict[str, Any]]] = None,
        const_additions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generates a diff preview for proposed additions without applying them.

        Returns:
            A string representing the diff.
        """
        temp_kb = self.knowledge_base.copy()

        if kb_additions:
            if "knowledge_base" not in temp_kb:
                temp_kb["knowledge_base"] = []
            temp_kb["knowledge_base"].extend(kb_additions)

        if func_additions:
            if "functions" not in temp_kb:
                temp_kb["functions"] = []
            temp_kb["functions"].extend(func_additions)

        if const_additions:
            if "constants" not in temp_kb:
                temp_kb["constants"] = {}
            for category, data in const_additions.items():
                if category not in temp_kb["constants"]:
                    temp_kb["constants"][category] = {"sort": data["sort"], "members": []}
                temp_kb["constants"][category]["members"].extend(data["members"])

        original_content = json.dumps(self.knowledge_base, indent=2).splitlines(keepends=True)
        proposed_content = json.dumps(temp_kb, indent=2).splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_content,
            proposed_content,
            fromfile="current_knowledge_base.json",
            tofile="proposed_knowledge_base.json",
            lineterm="",
        )
        return "".join(diff)

    def apply_additions(
        self,
        kb_additions: Optional[List[str]] = None,
        func_additions: Optional[List[Dict[str, Any]]] = None,
        const_additions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Applies proposed additions to the knowledge base and saves it."""
        if kb_additions:
            if "knowledge_base" not in self.knowledge_base:
                self.knowledge_base["knowledge_base"] = []
            self.knowledge_base["knowledge_base"].extend(kb_additions)

        if func_additions:
            if "functions" not in self.knowledge_base:
                self.knowledge_base["functions"] = []
            self.knowledge_base["functions"].extend(func_additions)

        if const_additions:
            if "constants" not in self.knowledge_base:
                self.knowledge_base["constants"] = {}
            for category, data in const_additions.items():
                if category not in self.knowledge_base["constants"]:
                    self.knowledge_base["constants"][category] = {"sort": data["sort"], "members": []}
                self.knowledge_base["constants"][category]["members"].extend(data["members"])

        self._save_knowledge_base()
        logger.info("Knowledge base updated with new additions.")
