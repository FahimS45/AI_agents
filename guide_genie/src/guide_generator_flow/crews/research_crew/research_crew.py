from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import (
    YoutubeVideoSearchTool,
    YoutubeChannelSearchTool,
    ScrapeWebsiteTool,
    SeleniumScrapingTool,
    ArxivPaperTool,
    PDFSearchTool,
    CodeDocsSearchTool,
    TXTSearchTool,
    MDXSearchTool

)
from guide_generator_flow.tools.youtube_transcription_tool import YoutubeTranscriptionTool

yt_video_search_tool = YoutubeVideoSearchTool()
yt_channel_search_tool = YoutubeChannelSearchTool()
yt_transcription_tool = YoutubeTranscriptionTool()

web_scraping_tool = ScrapeWebsiteTool()
selenium_scraping_tool = SeleniumScrapingTool()
code_docs_search_tool = CodeDocsSearchTool()

arxiv_tool = ArxivPaperTool()

pdf_search_tool = PDFSearchTool()
txt_search_tool = TXTSearchTool()
md_search_tool = MDXSearchTool()

@CrewBase
class ResearchCrew():
    """ResearchCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['research_manager'] # type: ignore[index]
        )
    
    @agent
    def youtube_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["youtube_specialist"],
            tools=[yt_channel_search_tool, yt_video_search_tool],
            max_iter=5 
        )

    @agent
    def web_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["web_specialist"],
            tools=[web_scraping_tool, selenium_scraping_tool, code_docs_search_tool],
            max_iter=5
        )
    
    @agent
    def arxiv_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["arxiv_specialist"],
            tools=[arxiv_tool],
            max_iter=5
        )
    
    @agent
    def document_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["document_specialist"],
            tools=[pdf_search_tool, txt_search_tool, md_search_tool],
            max_iter=5
        )
    
    @task
    def research_compilation(self) -> Task:
        return Task(
            config=self.tasks_config["research_compilation"]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.youtube_specialist(),
                    self.web_specialist(),
                    self.arxiv_specialist(),
                    self.document_specialist()],
            tasks=[self.research_compilation()],
            verbose=True,
            process=Process.hierarchical,
            planning=True,
            manager_agent=self.research_manager()
        )
    
def run(input: dict[str, str]) -> str:
    
    research_crew = ResearchCrew().crew()
    result = research_crew.kickoff(input)

    return result.raw

if __name__ == "__main__":
    inputs={
        "youtube_links": "",
        "webpage_links": "https://medium.com/offnote-labs/a-generalized-approach-to-virtual-try-on-245e64779f18",
        "research_paper_links": "",
        "document_paths": ""
    }

    result = run(inputs)
    print(result)