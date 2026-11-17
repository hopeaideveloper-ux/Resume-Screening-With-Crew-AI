# AI Resume Screening — Flask + CrewAI + OpenAI

A beginner-friendly example using **Agents, Tasks, a sequential Crew, a CrewAI Flow, task guardrails, and Flow persistence**.

## 1. Project flow

1. Flask lists supported resumes and job descriptions in their respective folders.
2. The user selects one job description, and the app extracts text from every PDF, DOCX, or TXT resume.
3. A CrewAI Flow validates each resume and the selected job description.
4. The sequential Crew runs three agents for each resume:
   - Resume Evidence Analyst — extracts factual evidence.
   - Job Requirements Analyst — compares evidence with the job description.
   - Fairness and Quality Reviewer — checks unsupported claims and bias.
5. A built-in Task guardrail checks the final task output and can request a retry.
6. The Flow stores its state using CrewAI's default SQLite persistence.
7. Flask displays extracted details and requirement evidence in a comparison table, with a match score for each resume.

## 2. Input files

Put resume files in `resume` and job-description files in `job_description`. The page lists PDF, DOCX, and TXT files in each folder. Select one job description to compare against every supported resume in `resume`. Add or remove files in those folders and refresh the page to update the choices.

The score weights each distinct job requirement equally: `met` earns 1 point, `partial` earns 0.5, and `not_evidenced` earns 0. The score is the percentage of available points, rounded to the nearest whole number. It is an indicative evidence aid, not a probability or hiring decision; missing evidence is not proof that a candidate lacks a qualification.

## 3. Setup (Windows)

```bash
py -m venv venv
venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_real_key
```

Run:

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## 4. CrewAI built-ins used

| Built-in | Where | Purpose |
|---|---|---|
| Agent | `resume_crew.py` | Role, goal, backstory, LLM, iteration limit |
| Task | `resume_crew.py` | Instructions, expected output, task dependencies |
| `context` | Matching and review tasks | Pass earlier task outputs forward |
| `Process.sequential` | Crew | Run tasks in a predictable order |
| Task `guardrail` | Final review task | Validate output; allow limited retries |
| Flow `@start` / `@listen` | Flow class | Orchestrate steps and expose workflow checkpoints |
| Flow `@persist` | Flow class | Persist Flow state in SQLite |
| `Flow.state.id` | Flow state | Identify a persisted Flow execution |

### Important distinction: guardrails and checkpoints

CrewAI's built-in `guardrail` here is attached to a **Task**, not to the `Crew` constructor. The app also performs ordinary input validation before starting the Flow. The `@listen` method is a workflow checkpoint, while `@persist` saves Flow state. Persistence is not the same as a full transactional job queue, and it does not guarantee that an interrupted OpenAI request will continue from the exact token where it stopped.

## 5. Change the model

Set `OPENAI_MODEL` in `.env` to a model available to your OpenAI API account. The default is `gpt-4o-mini`. The API key is read from the environment; it is not written in the Python source.

## 6. Beginner notes

- `resume_crew.py` contains the CrewAI logic.
- `app.py` contains the Flask routes and reads selected files from the input folders.
- `templates/index.html` contains the page.
- `static/style.css` contains the page styling.
- `flow_states.db` (created by CrewAI in its configured storage location) stores persisted Flow state. Treat it as sensitive because it can contain resume text. Protect it, restrict access, and apply a retention/deletion policy.

## 7. Responsible-use and security notes

- This is a learning/demo application, not a validated hiring product.
- The report is decision support only. A human recruiter must verify all claims and make the hiring decision.
- Do not score or filter using protected traits or proxies such as names, photos, age, gender, religion, disability, ethnicity, or address.
- Obtain appropriate permission and follow applicable privacy, employment, and data-protection requirements before uploading resumes to an external AI service.
- Do not expose Flask debug mode publicly. Use HTTPS, authentication, rate limiting, secure secret management, malware scanning, and a production WSGI server before deployment.
- The app reads source files from the input folders and does not intentionally modify them. Flow persistence may still store extracted resume text; secure and delete the SQLite state store according to your retention policy.
- Add authentication and authorization before allowing multiple recruiters to use this application.

## 8. Troubleshooting

**`OPENAI_API_KEY` missing:** Check that `.env` exists in the same folder as `app.py`, and that the key is valid.

**A file is not listed:** Ensure it is in the correct folder and uses a PDF, DOCX, or TXT extension.

**PDF text is empty:** The PDF may be scanned. Add OCR (for example, Tesseract) before screening.

**Package/API mismatch:** CrewAI evolves. Upgrade the packages in the active virtual environment and consult the current CrewAI documentation before changing imports or Flow APIs.

**OpenAI model unavailable:** Set `OPENAI_MODEL` to a model enabled for your API project.
