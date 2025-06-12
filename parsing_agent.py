import json
import os
import datetime
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from utils import setup_logger

logger = setup_logger('parsing_agent', 'parsing_agent.log')

class ParsingAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["clarified_input"],
            template="""
You are given a clarified simulation specification intended for FEniCS.
Your task is to parse this paragraph into a structured JSON object that captures all key attributes needed for code generation.

📌 The output must be a valid JSON with the following fields:
- problem_type (e.g., heat, fluid, elasticity, hyperelasticity, fracture)
- pde_description (a short descriptive sentence)
- dimension (1, 2, or 3)
- domain (shape description)
- domain_geometry_file (optional, use null if not needed)
- mesh (object with fields: nx, ny[, nz]) — include nz only if 3D
- variables (e.g., ["u"], ["u", "p"], ["u", "d"])
- time_dependent (true/false)
- nonlinear (true/false)
- coupled (true/false)
- boundary_conditions (list of Dirichlet or Neumann conditions)
- initial_conditions (initial values for each variable)
- source_terms (list, or empty array [])
- material_properties (dictionary of physical parameters)
- notes (optional field for special considerations)

📦 Output only the JSON object. Do not include any explanation, markdown, or code block.

Input:
{clarified_input}

Output:
"""
        )
        self.chain = self.prompt_template | self.llm

    def parse(self, clarified_input: str) -> dict:
        logger.info(f"Parsing clarified input: {clarified_input}")
        response = None
        try:
            # LLM에 clarified_input을 보내서 파싱 시도
            response = self.chain.invoke({"clarified_input": clarified_input})
            content = response.content.strip()

            # 이상한 코드 블록 formatting 제거
            if content.startswith("```"):
                content = content.split("```")[-1].strip()

            parsed_fields = json.loads(content)

            # 최종 출력은 parsed + full_text 모두 포함
            parsed_data = {
                "parsed": parsed_fields,
                "full_text": clarified_input
            }

            self.save_parsed_json(parsed_data)
            self.save_parsed_text(parsed_data)

        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            fallback = response.content if response and hasattr(response, "content") else str(clarified_input)
            parsed_data = {
                "parsed": {"fallback_text": fallback},
                "full_text": clarified_input
            }

        logger.info(f"Parsed data: {parsed_data}")
        return parsed_data

    def save_parsed_json(self, parsed_data: dict, path: str = "parsed_results.jsonl"):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(parsed_data, ensure_ascii=False) + "\n")
            logger.info(f"✅ JSON saved to {path}")
        except Exception as e:
            logger.error(f"❌ Failed to save JSON: {str(e)}")

    def save_parsed_text(self, parsed_data: dict, path: str = "parsed_results.txt"):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write("[PARSED RESULT]\n")
                for key, value in parsed_data.items():
                    f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
                f.write("\n" + "-" * 80 + "\n\n")
            logger.info(f"✅ Text log saved to {path}")
        except Exception as e:
            logger.error(f"❌ Failed to save text log: {str(e)}")
