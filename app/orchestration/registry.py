from app.models.agent import BaseAgent


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}

    def register(self, name: str, agent: BaseAgent) -> None:
        self._agents[name] = agent

    def get(self, name: str) -> BaseAgent:
        if name not in self._agents:
            raise KeyError(f"Unknown agent: {name}")
        return self._agents[name]

    def list(self) -> list[str]:
        return list(self._agents.keys())

    def health(self) -> dict[str, bool]:
        return {name: agent.health() for name, agent in self._agents.items()}
