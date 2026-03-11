import json
import os
import typing
from datetime import datetime
from enum import Enum


class NodeType(Enum):
    CHAT = 1
    THOUGHT = 2
    EVENT = 3
    ACTION = 4

    def toJSON(self):
        return self.name


class Node:
    id: int
    type: NodeType

    subject: str
    predicate: str
    object: str

    description: str

    importance_score: float

    created: datetime
    expiration: datetime

    always_include: bool

    def __init__(
        self,
        id: int,
        type: NodeType,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        self.id = id
        self.type = type
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.description = description
        self.created = created
        self.expiration = expiration
        self.always_include = always_include

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    def toJSON(self):
        return {
            "id": self.id,
            "type": self.type.toJSON(),
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "importance_score": self.importance_score,
            "created": self.created.strftime("%Y-%m-%d %H:%M:%S"),
            "expiration": self.expiration.strftime("%Y-%m-%d %H:%M:%S"),
            "always_include": "true" if self.always_include else "false",
        }


class Thought(Node):
    def __init__(
        self,
        id: int,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        super().__init__(
            id,
            NodeType.THOUGHT,
            subject,
            predicate,
            object,
            description,
            created,
            expiration,
            always_include,
        )


class Chat(Node):
    conversation: list[tuple[str, str]]

    def __init__(
        self,
        id: int,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        self.conversation = []
        super().__init__(
            id,
            NodeType.CHAT,
            subject,
            predicate,
            object,
            description,
            created,
            expiration,
            always_include,
        )

    def toJSON(self):
        return {
            "id": self.id,
            "type": self.type.toJSON(),
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "importance_score": self.importance_score,
            "conversation": self.conversation,
            "created": self.created.strftime("%Y-%m-%d %H:%M:%S"),
            "expiration": self.expiration.strftime("%Y-%m-%d %H:%M:%S"),
            "always_include": "true" if self.always_include else "false",
        }


class Event(Node):
    def __init__(
        self,
        id: int,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        super().__init__(
            id,
            NodeType.EVENT,
            subject,
            predicate,
            object,
            description,
            created,
            expiration,
            always_include,
        )


class Action(Node):
    def __init__(
        self,
        id: int,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        super().__init__(
            id,
            NodeType.ACTION,
            subject,
            predicate,
            object,
            description,
            created,
            expiration,
            always_include,
        )


class AssociativeMemory:
    def __init__(self, base_path, do_load=False) -> None:
        self.id_to_node: typing.Dict[int, Node] = dict()

        self.thought_id_to_node: typing.Dict[int, Thought] = dict()
        self.chat_id_to_node: typing.Dict[int, Node] = dict()
        self.event_id_to_node: typing.Dict[int, Node] = dict()
        self.action_id_to_node: typing.Dict[int, Node] = dict()

        self.nodes_without_chat_by_time: list[Node] = []

        self.base_path = base_path
        self.init_memory_md()
        if os.path.exists(f"{base_path}/nodes.json") and do_load:
            self._load(base_path)

    def init_memory_md(self):
        memory_md_path = f"{self.base_path}/MEMORY.md"
        if not os.path.exists(memory_md_path):
            with open(memory_md_path, "w") as f:
                f.write("# Private Memory\n")

    def _load(self, base_path):
        raise NotImplementedError("AssociativeMemory loading is not implemented.")

    def save(self):
        with open(f"{self.base_path}/nodes.json", "w") as f:
            json.dump([node.toJSON() for node in self.id_to_node.values()], f)

    def _add(
        self, subject, predicate, obj, description, type, created, expiration
    ) -> Node:
        id = len(self.id_to_node) + 1

        if type == NodeType.CHAT:
            node = Chat(id, subject, predicate, obj, description, created, expiration)
            self.chat_id_to_node[id] = node
        elif type == NodeType.THOUGHT:
            node = Thought(
                id, subject, predicate, obj, description, created, expiration
            )
            self.thought_id_to_node[id] = node
        elif type == NodeType.EVENT:
            node = Event(id, subject, predicate, obj, description, created, expiration)
            self.event_id_to_node[id] = node
        elif type == NodeType.ACTION:
            node = Action(id, subject, predicate, obj, description, created, expiration)
            self.action_id_to_node[id] = node

        if type != NodeType.CHAT:
            self.nodes_without_chat_by_time.append(node)

        self.id_to_node[id] = node

        return node

    def add_chat(
        self, subject, predicate, obj, description, conversation, created, expiration
    ) -> Chat:
        node = self._add(
            subject, predicate, obj, description, NodeType.CHAT, created, expiration
        )
        node.conversation = conversation
        return node

    def add_thought(
        self, subject, predicate, obj, description, created, expiration
    ) -> Thought:
        return self._add(
            subject, predicate, obj, description, NodeType.THOUGHT, created, expiration
        )

    def add_event(
        self, subject, predicate, obj, description, created, expiration
    ) -> Event:
        return self._add(
            subject, predicate, obj, description, NodeType.EVENT, created, expiration
        )

    def add_action(
        self, subject, predicate, obj, description, created, expiration
    ) -> Action:
        return self._add(
            subject, predicate, obj, description, NodeType.ACTION, created, expiration
        )

    def get_nodes_for_retrieval(self, current_time: datetime) -> list[Node]:
        """
        Get all nodes except chat
        """
        nodes = []
        for node in self.nodes_without_chat_by_time:
            if node.expiration > current_time:
                nodes.append(node)
        return nodes

    def append_to_memory_md(self, node: Node):
        type_name = node.type.name.lower()
        description_lines = node.description.splitlines() or [""]

        with open(f"{self.base_path}/MEMORY.md", "a") as f:
            if os.path.getsize(f"{self.base_path}/MEMORY.md") > 0:
                f.write("\n")
            f.write(
                "- "
                f"{node.created.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"type={type_name} | "
                f"importance={node.importance_score:g} | "
                f"expires={node.expiration.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"always_include={'true' if node.always_include else 'false'}\n"
            )
            for line in description_lines:
                f.write(f"  {line}\n")

    def read_memory_md(
        self, current_time: datetime
    ) -> list[tuple[datetime, str, float, bool]]:
        memory_md_path = f"{self.base_path}/MEMORY.md"
        if not os.path.exists(memory_md_path):
            self.init_memory_md()
            return []

        def parse_entry(header: str, description_lines: list[str]):
            if not header.startswith("- "):
                return None

            parts = [part.strip() for part in header[2:].split(" | ")]
            if len(parts) < 5:
                return None

            created = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            fields = {}
            for part in parts[1:]:
                if "=" not in part:
                    return None
                key, value = part.split("=", 1)
                fields[key] = value

            if fields.get("type") == "chat":
                return None

            expires = datetime.strptime(fields["expires"], "%Y-%m-%d %H:%M:%S")
            if expires <= current_time:
                return None

            description = "\n".join(description_lines).strip()
            importance = float(fields["importance"])
            always_include = fields["always_include"].lower() == "true"
            return (created, description, importance, always_include)

        entries: list[tuple[datetime, str, float, bool]] = []
        current_header = None
        current_description_lines: list[str] = []

        with open(memory_md_path, "r") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")

                if line.startswith("# "):
                    continue
                if line.startswith("- "):
                    if current_header is not None:
                        entry = parse_entry(current_header, current_description_lines)
                        if entry is not None:
                            entries.append(entry)
                    current_header = line
                    current_description_lines = []
                    continue
                if current_header is None:
                    continue

                current_description_lines.append(
                    line[2:] if line.startswith("  ") else line
                )

        if current_header is not None:
            entry = parse_entry(current_header, current_description_lines)
            if entry is not None:
                entries.append(entry)

        return entries
