"""AI Study Pack Generator.

Run in Colab/locally (Gradio):  python app.py
Run for deployment (Streamlit): streamlit run app.py
"""

import json
import os
import re
from typing import Any

from groq import Groq
from json_repair import repair_json


APP_TITLE = "AI Study Pack Generator"
COMPONENTS = [
    "Topic Summary",
    "Key Concepts",
    "Flashcards",
    "Multiple-Choice Quiz",
    "Short-Answer Questions",
    "Study Plan",
]
LANGUAGES = ["English", "Urdu", "Arabic", "Hindi", "French", "Spanish"]
LEVELS = ["School", "College", "Undergraduate", "Postgraduate", "Professional"]
DIFFICULTIES = ["Easy", "Moderate", "Advanced"]


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object even if the model adds a Markdown fence."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        candidate = match.group(0) if match else cleaned
        result = repair_json(candidate, return_objects=True, ensure_ascii=False)
    if not isinstance(result, dict) or not result:
        raise ValueError("The AI response could not be converted into a JSON object.")
    return result


def _prompt(
    topic: str,
    level: str,
    language: str,
    difficulty: str,
    notes: str,
    flashcard_count: int,
    mcq_count: int,
    short_count: int,
    plan_days: int,
    components: list[str],
) -> str:
    selected = ", ".join(components)
    source_rule = (
        "Use the provided notes as the primary source; do not contradict them."
        if notes.strip()
        else "Use accurate, well-established educational knowledge."
    )
    return f"""
You are an expert teacher and instructional designer. Create a clear, accurate,
age-appropriate study pack about: {topic}

Learner level: {level}
Output language: {language}
Difficulty: {difficulty}
Required components: {selected}
Number of flashcards: {flashcard_count}
Number of MCQs: {mcq_count}
Number of short-answer questions: {short_count}
Study-plan length: {plan_days} days
Teacher/lecture notes:
{notes.strip() or "No notes supplied."}

{source_rule}
Explain unfamiliar terms simply. Make questions assess understanding rather than
mere recall. Each MCQ must have exactly four options, one correct answer, and a
brief explanation. Each short-answer question must include a model answer.

Return ONLY valid JSON. Include only the requested component keys, using this schema:
{{
  "topic_summary": "string",
  "key_concepts": [{{"concept": "string", "explanation": "string"}}],
  "flashcards": [{{"front": "string", "back": "string"}}],
  "multiple_choice_quiz": [
    {{"question": "string", "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "A", "explanation": "string"}}
  ],
  "short_answer_questions": [{{"question": "string", "model_answer": "string"}}],
  "study_plan": [{{"day": 1, "focus": "string", "activities": ["string"]}}]
}}
""".strip()


def _to_markdown(data: dict[str, Any], topic: str) -> str:
    lines = [
        f"# Study Pack: {topic.strip()}",
        "*AI workflow completed: Planning → Content → Assessment → Review → Refinement*",
    ]

    if data.get("topic_summary"):
        lines += ["\n## Topic Summary", str(data["topic_summary"])]

    if data.get("key_concepts"):
        lines.append("\n## Key Concepts")
        for item in data["key_concepts"]:
            lines.append(f"- **{item.get('concept', '')}:** {item.get('explanation', '')}")

    if data.get("flashcards"):
        lines.append("\n## Flashcards")
        for number, card in enumerate(data["flashcards"], 1):
            lines += [f"\n### Card {number}", f"**Front:** {card.get('front', '')}",
                      f"**Back:** {card.get('back', '')}"]

    if data.get("multiple_choice_quiz"):
        lines.append("\n## Multiple-Choice Quiz")
        answers = []
        for number, item in enumerate(data["multiple_choice_quiz"], 1):
            lines.append(f"\n**{number}. {item.get('question', '')}**")
            lines.extend(str(option) for option in item.get("options", []))
            answers.append(
                f"{number}. **{item.get('answer', '')}** — {item.get('explanation', '')}"
            )
        lines += ["\n### Answer Key", *answers]

    if data.get("short_answer_questions"):
        lines.append("\n## Short-Answer Questions")
        for number, item in enumerate(data["short_answer_questions"], 1):
            lines += [f"\n**{number}. {item.get('question', '')}**",
                      f"*Model answer:* {item.get('model_answer', '')}"]

    if data.get("study_plan"):
        lines.append("\n## Study Plan")
        for item in data["study_plan"]:
            activities = "; ".join(map(str, item.get("activities", [])))
            lines.append(f"- **Day {item.get('day', '')}: {item.get('focus', '')}** — {activities}")

    return "\n".join(lines)


def generate_study_pack(
    topic: str,
    level: str,
    language: str,
    difficulty: str,
    notes: str,
    flashcard_count: int,
    mcq_count: int,
    short_count: int,
    plan_days: int,
    components: list[str],
    api_key: str = "",
    progress_callback=None,
) -> str:
    if not topic or not topic.strip():
        return "Please enter a study topic."
    if not components:
        return "Please select at least one study-pack component."

    key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if not key:
        return "GROQ_API_KEY is missing. Add it in Colab, Streamlit Secrets, or the API-key field."

    def update(stage: str, detail: str) -> None:
        if progress_callback:
            progress_callback(stage, detail)

    def call_stage(client: Groq, stage: str, instruction: str, context: dict[str, Any]) -> dict[str, Any]:
        """Run one workflow stage with JSON parsing, repair, and clear stage errors."""
        update(stage, "Running")
        compact_context = json.dumps(context, ensure_ascii=False)[:30000]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert instructional designer. Return only one valid JSON object, "
                    "without Markdown fences, comments, or text outside the JSON."
                ),
            },
            {"role": "user", "content": f"TASK:\n{instruction}\n\nCONTEXT:\n{compact_context}"},
        ]
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b", temperature=0.2, messages=messages
            )
            raw = response.choices[0].message.content or ""
            try:
                result = _extract_json(raw)
            except (ValueError, json.JSONDecodeError):
                repair = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "Convert the input into valid JSON only."},
                        {"role": "user", "content": raw[:30000]},
                    ],
                )
                result = _extract_json(repair.choices[0].message.content or "")
            update(stage, "Completed")
            return result
        except Exception as exc:
            update(stage, "Failed")
            raise RuntimeError(f"{stage} stage failed: {exc}") from exc

    try:
        client = Groq(api_key=key)
        request_context = {
            "topic": topic.strip(), "learner_level": level, "language": language,
            "difficulty": difficulty, "notes": notes.strip(), "components": components,
            "counts": {"flashcards": int(flashcard_count), "mcqs": int(mcq_count),
                       "short_answers": int(short_count), "study_days": int(plan_days)},
        }

        plan = call_stage(
            client, "1. Planning",
            "Create a learning design plan with keys: learning_objectives (array), scope (array), "
            "sequence (array), assessment_strategy (string), and source_guidance (string).",
            request_context,
        )
        content = call_stage(
            client, "2. Content generation",
            "Using the plan and user request, draft only the requested instructional components. "
            "Use keys topic_summary, key_concepts, flashcards, and study_plan following the original schema. "
            "Respect the requested counts and use notes as the primary source when supplied.",
            {"request": request_context, "plan": plan},
        )
        assessment = call_stage(
            client, "3. Assessment",
            "Create the requested multiple_choice_quiz and short_answer_questions. Each MCQ must have "
            "four options, one answer letter, and an explanation. Include model answers for short questions.",
            {"request": request_context, "plan": plan, "content": content},
        )
        review = call_stage(
            client, "4. Quality review",
            "Audit alignment, accuracy, clarity, difficulty, language, counts, answer correctness, duplication, "
            "and consistency with supplied notes. Return keys passed (boolean), issues (array), and improvements (array).",
            {"request": request_context, "plan": plan, "content": content, "assessment": assessment},
        )
        final_pack = call_stage(
            client, "5. Refinement",
            "Produce the corrected final study pack using the review. Return only requested keys from this schema: "
            "topic_summary; key_concepts[{concept, explanation}]; flashcards[{front, back}]; "
            "multiple_choice_quiz[{question, options, answer, explanation}]; "
            "short_answer_questions[{question, model_answer}]; study_plan[{day, focus, activities}]. "
            "Meet all requested counts exactly.",
            {"request": request_context, "plan": plan, "content": content,
             "assessment": assessment, "review": review},
        )
        return _to_markdown(final_pack, topic)
    except Exception as exc:
        return f"Generation failed: {exc}"


def run_gradio() -> None:
    import gradio as gr

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# 🎓 {APP_TITLE}\nCreate a personalized learning resource in seconds.")
        with gr.Row():
            with gr.Column():
                topic = gr.Textbox(label="Study topic", placeholder="e.g., Correlation and causation")
                level = gr.Dropdown(LEVELS, value="Undergraduate", label="Learner level")
                language = gr.Dropdown(LANGUAGES, value="English", label="Output language")
                difficulty = gr.Radio(DIFFICULTIES, value="Moderate", label="Difficulty")
                notes = gr.Textbox(label="Optional lecture/textbook notes", lines=7)
                components = gr.CheckboxGroup(COMPONENTS, value=COMPONENTS, label="Study-pack components")
                api_key = gr.Textbox(label="Groq API key (optional if stored as GROQ_API_KEY)", type="password")
            with gr.Column():
                flashcards = gr.Slider(3, 20, value=8, step=1, label="Flashcards")
                mcqs = gr.Slider(3, 20, value=8, step=1, label="MCQs")
                short_questions = gr.Slider(2, 10, value=5, step=1, label="Short-answer questions")
                days = gr.Slider(1, 14, value=7, step=1, label="Study-plan days")
                generate = gr.Button("Generate Study Pack", variant="primary")
        output = gr.Markdown(label="Generated study pack")
        generate.click(
            generate_study_pack,
            inputs=[topic, level, language, difficulty, notes, flashcards, mcqs,
                    short_questions, days, components, api_key],
            outputs=output,
        )
    demo.launch(share=True)


def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
    st.title(f"🎓 {APP_TITLE}")
    st.caption("Create a personalized summary, revision tools, assessment, and study plan.")

    try:
        stored_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        stored_key = ""

    with st.sidebar:
        st.header("Study settings")
        level = st.selectbox("Learner level", LEVELS, index=2)
        language = st.selectbox("Output language", LANGUAGES)
        difficulty = st.radio("Difficulty", DIFFICULTIES, index=1)
        flashcards = st.slider("Flashcards", 3, 20, 8)
        mcqs = st.slider("MCQs", 3, 20, 8)
        short_questions = st.slider("Short-answer questions", 2, 10, 5)
        days = st.slider("Study-plan days", 1, 14, 7)
        api_key = st.text_input(
            "Groq API key",
            type="password",
            help="Leave blank when GROQ_API_KEY is saved in Streamlit Secrets.",
        )

    topic = st.text_input("Study topic", placeholder="e.g., Correlation and causation")
    notes = st.text_area("Optional lecture/textbook notes", height=180)
    components = st.multiselect("Study-pack components", COMPONENTS, default=COMPONENTS)

    if st.button("Generate Study Pack", type="primary", use_container_width=True):
        with st.status("Running the AI workflow...", expanded=True) as workflow_status:
            stage_lines = {}

            def show_progress(stage: str, detail: str) -> None:
                stage_lines[stage] = detail
                icon = "✅" if detail == "Completed" else "❌" if detail == "Failed" else "⏳"
                st.write(f"{icon} {stage}: {detail}")

            result = generate_study_pack(
                topic, level, language, difficulty, notes, flashcards, mcqs,
                short_questions, days, components, api_key or stored_key,
                progress_callback=show_progress,
            )
            if result.startswith("Generation failed:"):
                workflow_status.update(label="Workflow stopped with an error", state="error")
            else:
                workflow_status.update(label="AI study-pack workflow completed", state="complete")
        st.session_state["study_pack"] = result

    if "study_pack" in st.session_state:
        st.markdown(st.session_state["study_pack"])
        st.download_button(
            "Download as Markdown",
            st.session_state["study_pack"],
            file_name="study_pack.md",
            mime="text/markdown",
        )


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_in_streamlit():
        run_streamlit()
    else:
        run_gradio()
