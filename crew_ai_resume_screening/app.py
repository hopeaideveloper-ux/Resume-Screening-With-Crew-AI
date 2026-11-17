import json
import os
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, flash
from dotenv import load_dotenv

from resume_crew import CandidateAssessment, ResumeScreeningFlow, extract_resume_text

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
RESUME_DIR = BASE_DIR / "resume"
JOB_DESCRIPTION_DIR = BASE_DIR / "job_description"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-before-deployment")

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def list_input_files(directory):
    if not directory.is_dir():
        return []
    return sorted(
        file.name
        for file in directory.iterdir()
        if file.is_file() and allowed_file(file.name)
    )


def calculate_match_score(assessment: CandidateAssessment) -> int | None:
    """Score equally weighted job requirements using explicit evidence statuses."""
    if not assessment.requirements:
        return None

    points = sum(
        1 if item.status == "met" else 0.5 if item.status == "partial" else 0
        for item in assessment.requirements
    )
    return round(points * 100 / len(assessment.requirements))


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    resumes = list_input_files(RESUME_DIR)
    job_descriptions = list_input_files(JOB_DESCRIPTION_DIR)
    selected_job_description = ""

    if request.method == "POST":
        selected_job_description = request.form.get("job_description", "")

        if selected_job_description not in job_descriptions:
            flash("Please select a job description from the job_description folder.", "error")
        elif not resumes:
            flash("No supported resumes were found in the resume folder.", "error")
        else:
            try:
                job_description = extract_resume_text(
                    BytesIO((JOB_DESCRIPTION_DIR / selected_job_description).read_bytes()),
                    selected_job_description,
                )

                if not job_description.strip():
                    raise ValueError("Not enough readable text was found in the job description.")

                results = []
                for filename in resumes:
                    try:
                        resume_text = extract_resume_text(
                            BytesIO((RESUME_DIR / filename).read_bytes()),
                            filename,
                        )
                        if len(resume_text.strip()) < 80:
                            raise ValueError("Not enough readable text was found in the resume.")

                        flow = ResumeScreeningFlow()
                        flow.state.job_description = job_description
                        flow.state.resume_text = resume_text
                        flow.kickoff()
                        assessment = CandidateAssessment(
                            **json.loads(flow.state.screening_result)
                        )

                        results.append({
                            "filename": filename,
                            "assessment": assessment,
                            "match_score": calculate_match_score(assessment),
                            "error": None,
                        })
                    except Exception as exc:
                        app.logger.exception("Resume screening failed for %s", filename)
                        results.append({
                            "filename": filename,
                            "assessment": None,
                            "match_score": None,
                            "error": str(exc),
                        })
            except Exception as exc:
                app.logger.exception("Job description processing failed")
                flash(f"Screening failed: {exc}", "error")

    return render_template(
        "index.html",
        results=results,
        resumes=resumes,
        job_descriptions=job_descriptions,
        selected_job_description=selected_job_description,
    )


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
