import json
import os
import datetime
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from utils import setup_logger

logger = setup_logger('error_diagnosis_agent', 'error_diagnosis_agent.log')
ERROR_LOG_FILE = "error_logs.txt"

def save_error_log(error_message: str, code: str, simulation_output: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"Timestamp: {timestamp}\n"
        f"Error Message: {error_message}\n"
        + ("-" * 80 + "\n") * 4 +
        f"Code:\n{code}\n"
        + ("-" * 80 + "\n") * 4 +
        f"Simulation Output:\n{simulation_output}\n"
        + ("-" * 80 + "\n") * 4
    )
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)
    logger.info("Error log saved to " + ERROR_LOG_FILE)

class ErrorDiagnosisAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)

    def diagnose_and_fix(self, error_message: str, code: str, simulation_output: str, iteration: int = -1) -> dict:
        logger.info("Starting error diagnosis...")
        save_error_log(error_message, code, simulation_output)

        if not simulation_output.strip():
            simulation_output = "[No output detected. The code may have failed to execute properly.]"
            logger.warning("Simulation output is empty; inserted placeholder message.")

        prompt = f"""
You are a diagnostic agent specialized in identifying and resolving errors in Python code for FEniCS-based simulations.

Your task is to:
1. Analyze the simulation output and error logs.
2. Accurately identify the root cause of failure in the original code.
3. Return the corrected full version of the code as plain Python (no markdown).

📦 Output Format (must follow this JSON schema):
{{
  "fix_type": "parsing" or "code",
  "hint": "Explanation of the error and how it was fixed",
  "after_code": "Modified full Python code",
  "confidence": float between 0.0 and 1.0
}}

🧾 Input Data:
[Error Message]
{error_message}

[Simulation Output Log]
{simulation_output}

[Original Code]
{code}
"""

        with open("last_prompt.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(prompt + "\n")

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            if content.startswith("```"):
                content = content.strip("`").replace("json", "").strip()

            result = json.loads(content)

            if not all(k in result for k in ["fix_type", "after_code"]):
                raise ValueError("Missing required keys in response JSON.")

            result["before_code"] = code
            if "diagnoses" not in result:
                result["diagnoses"] = [{"error_line": "unknown", "hint": "Automatically diagnosed."}]
            if "confidence" not in result:
                result["confidence"] = 0.5

            return result

        except Exception as e:
            logger.error(f"Error diagnosis failed: {e}")
            return {
                "fix_type": "code",
                "diagnoses": [{"error_line": "N/A", "hint": f"Diagnosis failed: {e}"}],
                "before_code": code,
                "after_code": code,
                "confidence": 0.0
            }
