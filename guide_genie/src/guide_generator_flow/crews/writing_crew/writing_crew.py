from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class WritingCrew():
    """WritingCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def technical_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_writer'], # type: ignore[index]
        )

    @agent
    def content_editor(self) -> Agent:
        return Agent(
            config=self.agents_config["content_editor"] # type: ignore[index] 
        )

    @task
    def write_getting_started_guide(self) -> Task:
        return Task(
            config=self.tasks_config['write_getting_started_guide'], # type: ignore[index]
        )

    @task
    def review_and_polish_guide(self) -> Task:
        return Task(
            config=self.tasks_config['review_and_polish_guide'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the WritingCrew crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )