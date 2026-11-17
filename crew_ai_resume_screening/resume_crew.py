import json
import importlib
import os
from io import BytesIO
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

_crewai = importlib.import_module("crewai")
Agent = _crewai.Agent
Crew = _crewai.Crew
Process = _crewai.Process
Task = _crewai.Task
LLM = _crewai.LLM
TaskOutput = _crewai.TaskOutput

_crewai_flow = importlib.import_module("crewai.flow")
Flow = _crewai_flow.Flow
start = _crewai_flow.start
listen = _crewai_flow.listen



class ScreeningState(BaseModel):
    job_description: str = ""
    resume_text: str = ""
    screening_result: str = ""


class RequirementAssessment(BaseModel):
    requirement: str = Field(
        description="One distinct, role-related requirement from the job description."
    )
    status: Literal["met", "partial", "not_evidenced"]
    evidence: str = Field(
        description="Resume evidence supporting the status, or a note that evidence was not stated."
    )


class CandidateAssessment(BaseModel):
    relevant_experience: list[str] = Field(description="Explicit, role-related experience from the resume.")
    education: list[str] = Field(description="Explicit education and certifications from the resume.")
    skills: list[str] = Field(description="Explicit skills relevant to the job description.")
    requirements: list[RequirementAssessment] = Field(
        description="Distinct job requirements, each assessed against explicit resume evidence."
    )


def extract_resume_text(file_storage, filename: str) -> str:
    """Read a supported document without saving it to disk."""
    content = file_storage.read()
    extension = filename.rsplit(".", 1)[1].lower()

    if extension == "pdf":
        fitz = importlib.import_module("fitz")
        with fitz.open(stream=content, filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)

    if extension == "docx":
        document_module = importlib.import_module("docx")
        document = document_module.Document(BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if extension == "txt":
        return content.decode("utf-8", errors="replace")

    raise ValueError("Unsupported file type.")


def validate_report(output: TaskOutput) -> tuple[bool, Any]:
    """Require a complete structured assessment before returning a result."""
    report = str(output.raw or "").strip()
    try:
        CandidateAssessment(**json.loads(report))
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        return False, f"Return valid JSON matching the CandidateAssessment fields and requirement statuses: {exc}"
    return True, report


class ResumeScreeningCrew:
    """A small sequential crew: extract evidence, compare, then audit the result."""

    def __init__(self):
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm = LLM(
            model=f"openai/{model_name}",
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        )

    def build(self):
        resume_reader = Agent(
            role="Resume Evidence Analyst",
            goal="Extract factual, job-relevant evidence from the resume without inventing details.",
            backstory=(
                "You are a careful resume analyst. Separate explicit evidence from missing "
                "information. Ignore protected characteristics and personal details."
            ),
            llm=self.llm,
            allow_delegation=False,
            max_iter=3,
            verbose=False,
        )

        job_matcher = Agent(
            role="Job Requirements Analyst",
            goal="Compare resume evidence with the supplied job requirements transparently.",
            backstory=(
                "You assess only role-related skills, experience, projects, and qualifications. "
                "You never infer ability from age, gender, name, photo, address, or other protected traits."
            ),
            llm=self.llm,
            allow_delegation=False,
            max_iter=3,
            verbose=False,
        )

        fairness_reviewer = Agent(
            role="Fairness and Quality Reviewer",
            goal="Audit the assessment for unsupported claims, bias, and unclear reasoning.",
            backstory=(
                "You are a cautious reviewer. Flag missing evidence and potential bias. "
                "Do not make the hiring decision."
            ),
            llm=self.llm,
            allow_delegation=False,
            max_iter=3,
            verbose=False,
        )

        evidence_task = Task(
            description=(
                "Read the resume below and extract only explicit, job-relevant evidence. "
                "Organize skills, work experience, education, projects, certifications, and "
                "measurable achievements. Mark absent information as 'Not stated'. "
                "Do not infer or fabricate.\n\nRESUME:\n{resume_text}"
            ),
            expected_output="A concise evidence inventory with facts and 'Not stated' gaps.",
            agent=resume_reader,
        )

        matching_task = Task(
            description=(
                "Compare the evidence inventory with the job description. Identify matched "
                "requirements, partial matches, and requirements with no resume evidence. "
                "For each distinct job-related requirement, set status to exactly 'met', "
                "'partial', or 'not_evidenced', and cite explicit resume evidence. Do not "
                "infer missing qualifications. Do not penalize career gaps or personal/"
                "protected characteristics.\n\nJOB DESCRIPTION:\n"
                "{job_description}"
            ),
            expected_output="A requirement-by-requirement analysis grounded in resume evidence.",
            agent=job_matcher,
            context=[evidence_task],
        )

        review_task = Task(
            description=(
                "Review the evidence inventory and requirement analysis for unsupported "
                "assumptions, invented facts, and protected-characteristic bias. Return only "
                "a valid JSON object, with no Markdown fences, containing these keys: "
                "relevant_experience (list of strings), education (list of strings), skills "
                "(list of strings), and requirements (list of objects with requirement, status, "
                "and evidence strings). Each status must be exactly 'met', 'partial', or "
                "'not_evidenced'. Use empty lists when a category has no stated information. "
                "Do not include personal or protected characteristics. Do not calculate a "
                "score; the application computes it from these statuses."
            ),
            expected_output=(
                'JSON matching {"relevant_experience": [], "education": [], "skills": [], '
                '"requirements": [{"requirement": "...", "status": "met", "evidence": "..."}]}'
            ),
            agent=fairness_reviewer,
            context=[evidence_task, matching_task],
            guardrail=validate_report,
            guardrail_max_retries=2,
        )

        return Crew(
            agents=[resume_reader, job_matcher, fairness_reviewer],
            tasks=[evidence_task, matching_task, review_task],
            process=Process.sequential,
            verbose=False,
        )


class ResumeScreeningFlow(Flow[ScreeningState]):
    """CrewAI Flow coordinates the run and persists state checkpoints in SQLite."""

    @start()
    def run_screening_crew(self):
        if not self.state.resume_text.strip() or not self.state.job_description.strip():
            raise ValueError("Resume text and job description are required.")

        crew = ResumeScreeningCrew().build()
        result = crew.kickoff(
            inputs={
                "resume_text": self.state.resume_text,
                "job_description": self.state.job_description,
            }
        )
        assessment = CandidateAssessment(**json.loads(str(result)))
        self.state.screening_result = json.dumps(assessment.model_dump())
        return self.state.screening_result

    @listen(run_screening_crew)
    def finish(self, report):
        # This listener is a named workflow checkpoint after the crew completes.
        self.state.screening_result = report
        return report
