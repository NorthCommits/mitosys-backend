from app.logger import get_logger

logger = get_logger("mitosys.registry")


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict = {}

    def add(self, name: str, agent) -> None:
        self._agents[name] = agent
        logger.info(f"Sub-agent '{name}' has been registered. Total active: {len(self._agents)}.")

    def remove(self, name: str) -> None:
        if name in self._agents:
            del self._agents[name]
            logger.info(
                f"Sub-agent '{name}' has been removed from the registry. "
                f"{len(self._agents)} sub-agent(s) remaining."
            )

    def get(self, name: str):
        return self._agents.get(name)

    def active_count(self) -> int:
        return len(self._agents)

    def active_names(self) -> list[str]:
        return list(self._agents.keys())
