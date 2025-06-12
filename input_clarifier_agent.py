from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger("input_clarifier")
logger.setLevel(logging.INFO)

class InputClarifierAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.2)
        self.prompt_template = PromptTemplate(
            input_variables=["raw_input"],
            template="""
You are given a user's simulation request in natural language. Based on the following guidelines, generate a **single-paragraph, fully specified simulation description** suitable for direct use in FEniCS.

📌 Carefully consider the following aspects when clarifying:
──────────────────────────────────────────────────────

🔷 Problem Type and Structure
- Identify the PDE type: heat, fluid (Navier-Stokes), elasticity, fracture, reaction-diffusion, etc.
- Specify whether the problem is steady or time-dependent.
- Determine if it is multiphysics (e.g., thermo-elasticity, fluid-structure interaction, electro-mechanics)..
- Identify spatial dimension: 2D or 3D.
- Describe the domain shape: rectangle, circle, cylinder, presence of notch, etc.

🔷 Field Variables and Conditions
- Define field variables such as u, p, T, d, k, etc.
- Infer and describe boundary and initial conditions clearly.
- Estimate required material properties: E, nu, k, rho, cp, mu, Gc, etc.

🔷 Numerical Settings
- Suggest appropriate time step dt (if transient).
- Recommend solver structure (e.g., nonlinear Newton solver, staggered scheme).
- Mention output format (e.g., whether to store results in .xdmf).

📦 Format your output as one complete paragraph in clear technical English.
📦 Do NOT include section headings or bullet points.

[User Request]
{raw_input}

[Refined Simulation Specification]
"""
        )

    def clarify(self, raw_input: str) -> str:
        try:
            prompt = self.prompt_template.format(raw_input=raw_input)
            logger.info("Clarifying input: %s", raw_input)
            result = self.llm.invoke(prompt)
            refined = result.content.strip()
            logger.info("Clarified result: %s", refined)
            return refined
        except Exception as e:
            logger.error("Clarification failed: %s", str(e))
            return raw_input  # fallback
