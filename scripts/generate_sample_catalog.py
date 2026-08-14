"""Generate a synthetic *multi-platform* sample catalog for the demo/tests.

Writes three CSVs in the **native schema** of each real Kaggle dataset, so the
multi-platform ingestion path (src/data/sources.py) is exercised end-to-end even
without the real files:

    examples/catalog/coursera.csv   (Coursera schema)
    examples/catalog/udemy.csv      (andrewmvd/udemy-courses schema)
    examples/catalog/edx.csv        (imuhammad/edx-courses schema)

The catalog spans concrete IT career tracks — Data Analyst, Data Engineer, ML
Engineer, MLOps/DevOps, Cloud, Product/Project Management, QA/Test Automation,
Web, Security, NLP/GenAI — so recommendations and (later) role roadmaps have
real coverage. Deterministic: same seed -> identical files.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

OUT_DIR = Path("examples/catalog")
SEED = 42

# track -> (skills pool, coarse category, edX subject, udemy subject)
TRACKS: dict[str, dict] = {
    "Data Analysis": {
        "skills": ["sql", "excel", "tableau", "power bi", "statistics", "data visualization", "pandas", "reporting"],
        "coursera_cat": "Data Analysis", "edx_subject": "Data Analysis & Statistics", "udemy_subject": "Business",
        "topics": ["Data Analytics", "SQL for Analysts", "Business Intelligence", "Data Visualization", "Spreadsheet Analytics"],
    },
    "Data Engineering": {
        "skills": ["spark", "airflow", "kafka", "etl", "data warehouse", "dbt", "snowflake", "sql"],
        "coursera_cat": "Data Engineering", "edx_subject": "Computer Science", "udemy_subject": "IT & Software",
        "topics": ["Data Engineering", "Big Data with Spark", "Building Data Pipelines", "Data Warehousing", "Streaming with Kafka"],
    },
    "Machine Learning": {
        "skills": ["python", "scikit-learn", "tensorflow", "pytorch", "deep learning", "model deployment", "feature engineering"],
        "coursera_cat": "Machine Learning", "edx_subject": "Computer Science", "udemy_subject": "Development",
        "topics": ["Machine Learning", "Deep Learning", "Applied ML", "Neural Networks", "ML Engineering"],
    },
    "MLOps & DevOps": {
        "skills": ["docker", "kubernetes", "ci/cd", "terraform", "monitoring", "github actions", "mlflow"],
        "coursera_cat": "Cloud Computing", "edx_subject": "Computer Science", "udemy_subject": "IT & Software",
        "topics": ["MLOps", "DevOps", "Kubernetes in Production", "CI/CD Pipelines", "Infrastructure as Code"],
    },
    "Cloud": {
        "skills": ["aws", "azure", "gcp", "cloud architecture", "serverless", "lambda", "networking"],
        "coursera_cat": "Cloud Computing", "edx_subject": "Computer Science", "udemy_subject": "IT & Software",
        "topics": ["AWS Cloud", "Cloud Architecture", "Serverless Applications", "Azure Fundamentals", "Cloud Security"],
    },
    "Product & Project Management": {
        "skills": ["roadmap", "stakeholder management", "agile", "scrum", "product strategy", "metrics", "prioritization"],
        "coursera_cat": "Business", "edx_subject": "Business & Management", "udemy_subject": "Business",
        "topics": ["Product Management", "Project Management", "Agile & Scrum", "Product Strategy", "Roadmapping"],
    },
    "QA & Test Automation": {
        "skills": ["selenium", "cypress", "test automation", "pytest", "qa", "test planning", "playwright"],
        "coursera_cat": "Software Development", "edx_subject": "Computer Science", "udemy_subject": "Development",
        "topics": ["Test Automation", "QA Engineering", "Selenium WebDriver", "API Testing", "End-to-End Testing"],
    },
    "Web Development": {
        "skills": ["react", "javascript", "typescript", "html", "css", "node.js", "rest api"],
        "coursera_cat": "Software Development", "edx_subject": "Computer Science", "udemy_subject": "Web Development",
        "topics": ["Web Development", "React", "Full-Stack JavaScript", "Frontend Engineering", "Node.js APIs"],
    },
    "Security": {
        "skills": ["cybersecurity", "cryptography", "penetration testing", "network security", "incident response"],
        "coursera_cat": "Security", "edx_subject": "Computer Science", "udemy_subject": "IT & Software",
        "topics": ["Cybersecurity", "Ethical Hacking", "Network Security", "Security Operations", "Cryptography"],
    },
    "NLP & GenAI": {
        "skills": ["nlp", "transformers", "llm", "rag", "prompt engineering", "hugging face", "embeddings"],
        "coursera_cat": "Machine Learning", "edx_subject": "Computer Science", "udemy_subject": "Development",
        "topics": ["Natural Language Processing", "Large Language Models", "Generative AI", "Building RAG Apps", "Transformers"],
    },
}

LEVELS = ["Beginner", "Intermediate", "Advanced"]
COURSERA_PARTNERS = ["Stanford University", "DeepLearning.AI", "Google", "IBM", "University of Michigan",
                     "Duke University", "Meta", "Imperial College London", "University of Colorado"]
EDX_INSTITUTIONS = ["HarvardX", "MITx", "IBM", "Microsoft", "GTx", "BerkeleyX", "UBCx", "DelftX"]
EDX_LEVELS = {"Beginner": "Introductory", "Intermediate": "Intermediate", "Advanced": "Advanced"}
UDEMY_LEVELS = {"Beginner": "Beginner Level", "Intermediate": "Intermediate Level", "Advanced": "Expert Level"}


def generate() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    coursera, udemy, edx = [], [], []
    for track, spec in TRACKS.items():
        for topic in spec["topics"]:
            for level in LEVELS:
                skills = ", ".join(rng.sample(spec["skills"], k=min(4, len(spec["skills"]))))

                # Coursera row
                partner = rng.choice(COURSERA_PARTNERS)
                coursera.append({
                    "course": f"{topic} {rng.choice(['Specialization', 'Professional Certificate', 'Course'])}",
                    "partner": partner,
                    "skills": skills,
                    "certificatetype": spec["coursera_cat"],
                    "level": level,
                    "rating": round(rng.uniform(4.1, 4.9), 1),
                })

                # Udemy row (no rating column; native schema)
                udemy.append({
                    "course_id": len(udemy) + 1000,
                    "course_title": f"The Complete {topic} Bootcamp",
                    # No fabricated course URL — the serving layer resolves a real
                    # provider search deep-link from the title (see src/recsys/links.py).
                    "url": "",
                    "is_paid": "True",
                    "price": rng.choice([0, 1999, 2999, 4999]),
                    "num_subscribers": rng.randint(500, 90000),
                    "num_reviews": rng.randint(20, 6000),
                    "num_lectures": rng.randint(20, 320),
                    "level": UDEMY_LEVELS[level],
                    "content_duration": round(rng.uniform(2, 42), 1),
                    "published_timestamp": f"20{rng.randint(18, 24)}-0{rng.randint(1, 9)}-15T00:00:00Z",
                    "subject": spec["udemy_subject"],
                })

                # edX row (rich: description + syllabus)
                inst = rng.choice(EDX_INSTITUTIONS)
                desc = (f"{topic} for {track.lower()}. Build practical, job-ready skills in "
                        f"{skills}. Designed for {level.lower()} learners aiming for a {track} role.")
                syllabus = " | ".join([
                    f"Module 1: Foundations of {topic}",
                    f"Module 2: Core tools ({skills.split(',')[0].strip()}, {skills.split(',')[-1].strip()})",
                    f"Module 3: Hands-on {topic} project",
                    f"Module 4: {level} techniques and best practices",
                    f"Module 5: Career path toward {track}",
                ])
                edx.append({
                    "title": f"{topic}",
                    "summary": f"A {level.lower()} path into {track}.",
                    "n_enrolled": rng.randint(1000, 200000),
                    "course_type": "Self-paced",
                    "institution": inst,
                    "instructors": "Course Staff",
                    "Level": EDX_LEVELS[level],
                    "subject": spec["edx_subject"],
                    "language": "English",
                    "subtitles": "English",
                    "course_effort": f"{rng.randint(2, 8)}–{rng.randint(9, 14)} hours per week",
                    "course_length": f"{rng.randint(4, 12)} Weeks",
                    "price": rng.choice(["Free", "Free (Audit)", "$49", "$99"]),
                    "course_description": desc,
                    "course_syllabus": syllabus,
                    "course_url": "",  # resolved to a real search deep-link at serve time
                })

    _write(OUT_DIR / "coursera.csv", coursera)
    _write(OUT_DIR / "udemy.csv", udemy)
    _write(OUT_DIR / "edx.csv", edx)
    print(f"Wrote sample catalog to {OUT_DIR}/ — "
          f"coursera={len(coursera)}, udemy={len(udemy)}, edx={len(edx)}")


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    generate()
