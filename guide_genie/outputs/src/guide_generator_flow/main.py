from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from  guide_generator_flow.crews.research_crew.research_crew import ResearchCrew
from guide_generator_flow.crews.writing_crew.writing_crew import WritingCrew
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()

class ResearchFlowState(BaseModel):

    youtube_links: Optional[str] = ""
    document_paths: Optional[str] = ""
    webpage_links: Optional[str] = ""
    research_paper_links: Optional[str] = ""

    research_report: str | None = None
    final_guide: str | None = None

class GuideGeneratorFlow(Flow[ResearchFlowState]):

    @start()
    def user_input(self) -> str:
        
        print("=" * 70)
        print("🚀 GUIDE GENERATOR FLOW STARTED")
        print("=" * 70)

        sources_provided = []

        if self.state.youtube_links:
            sources_provided.append("YouTube")
        elif self.state.document_paths:
            sources_provided.append("Documents")
        elif self.state.webpage_links:
            sources_provided.append("Webpages")
        elif self.state.research_paper_links:
            sources_provided.append("Research papers")
        
        if not sources_provided:
            print("\n⚠️  WARNING: No sources provided!")
            print("Please provide at least one source type.")
            return "no_sources"

        print(f"\n✅ Sources provided: {', '.join(sources_provided)}")
        print("\n" + "=" * 70)        

        return "input_received"
    
    @listen(user_input)
    def run_research_crew(self, prev_output) -> str:
        
        if prev_output == "no_sources":
            print("\n❌ Skipping research crew - no sources provided")
            return "research_skipped"
        
        elif prev_output == "input_received":
            print("\n" + "=" * 70)
            print("📚 CREW 1: RESEARCH CREW (Hierarchical)")
            print("=" * 70)
            print("\nInitializing research crew with manager + 4 specialists...")
            print("- YouTube Specialist")
            print("- Web Content Specialist")
            print("- Academic Paper Specialist")
            print("- Document Specialist")

            try:
                research_crew = ResearchCrew().crew()
                print("\n🔄 Delegating research tasks to specialists...\n")

                result = research_crew.kickoff(inputs={
                    "youtube_links": self.state.youtube_links or "Not provided",
                    "document_paths": self.state.document_paths or "Not provided",
                    "webpage_links": self.state.webpage_links or "Not provided",
                    "research_paper_links": self.state.research_paper_links or "Not provided"
                })

                self.state.research_report = result.raw
                
                print("\n" + "=" * 70)
                print("✅ RESEARCH CREW COMPLETED")
                print("=" * 70)
                print(f"📊 Research Report Generated:")

                return "research_complete"
            
            except Exception as e:
                print(f"\n❌ ERROR in Research Crew: {str(e)}")
                return "research_failed"

    @listen(run_research_crew)
    def run_writing_crew(self, prev_output) -> str:

        if prev_output == "research_skipped":
            print("\n❌ Skipping writing crew - research was skipped")
            return "writing_skipped"
        
        elif prev_output == "research_failed":
            print("\n❌ Skipping writing crew - research failed")
            return "writing_skipped"
        
        elif prev_output == "research_complete":
            print("\n" + "=" * 70)
            print("✍️  CREW 2: WRITING CREW (Sequential)")
            print("=" * 70)
            print("\nInitializing writing crew...")
            print("- Technical Writer (Step 1)")
            print("- Content Editor (Step 2)")
            print("\n" + "=" * 70)

            try:
                writing_crew = WritingCrew().crew()

                print("\n🔄 Transforming research into beginner-friendly guide...\n")

                result = writing_crew.kickoff(inputs={
                    "research_report": self.state.research_report 
                })

                self.state.final_guide = result.raw

                print("\n" + "=" * 70)
                print("✅ WRITING CREW COMPLETED")
                print("=" * 70)
                print(f"📝 Getting Started Guide Generated:")
                print("\n" + "=" * 70)

                return "guide_complete"
            
            except Exception as e:
                print(f"\n❌ ERROR in Writing Crew: {str(e)}")
                return "writing_failed"
    
def get_inputs() -> dict[str, str]:

    print("\n" + "=" * 70)
    print("🎯 GUIDE GENERATOR - INPUT COLLECTION")
    print("=" * 70)
    print("\nWelcome! Let's create a getting-started guide for your framework/tool.")
    print("\nℹ️  All source inputs are OPTIONAL. You can skip any by pressing Enter.")
    print("=" * 70)
    
    # YouTube Links (Optional)
    print("\n" + "─" * 70)
    print("\n📺 YOUTUBE VIDEOS/CHANNELS")
    print("   You can provide:")
    print("   - Individual video URLs (e.g., https://youtube.com/watch?v=abc123)")
    print("   - Channel URLs (e.g., https://youtube.com/@channelname)")
    print("   - Multiple links separated by commas")
    youtube_links = input("\n Enter youtube links (or press Enter to skip): ").strip()

    # Web Page Links (Optional)
    print("\n" + "─" * 70)
    print("\n🌐 WEB PAGES/ARTICLES")
    print("   You can provide:")
    print("   - Documentation URLs")
    print("   - Blog posts or tutorials")
    print("   - Multiple links separated by commas")
    webpage_links = input("\n Enter web page URLs (or press Enter to skip): ").strip()

    # Research Papers (Optional)
    print("\n" + "─" * 70)
    print("\n📄 RESEARCH PAPERS (arXiv)")
    print("   You can provide:")
    print("   - arXiv URLs (e.g., https://arxiv.org/abs/2103.xxxxx)")
    print("   - Paper titles or arXiv IDs")
    print("   - Multiple entries separated by commas")
    research_paper_links = input("\n Enter research paper links/queries (or press Enter to skip): ").strip()
    
    # Documents (Optional)
    print("\n" + "─" * 70)
    print("\n📁 DOCUMENTS (PDF/Text/Markdown)")
    print("   You can provide:")
    print("   - Local file paths to PDFs")
    print("   - Text file paths (.txt)")
    print("   - Markdown file paths (.md, .mdx)")
    print("   - Multiple paths separated by commas")
    document_paths = input("\n Enter document paths (or press Enter to skip): ").strip()

    return {
        "youtube_links": youtube_links,
        "document_paths": document_paths,
        "webpage_links": webpage_links,
        "research_paper_links": research_paper_links
    }

def kickoff():

    inputs = get_inputs()

    flow = GuideGeneratorFlow()

    flow_result = flow.kickoff(inputs=inputs)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"\n{flow_result}")

if __name__ == "__main__":
    kickoff()