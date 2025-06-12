from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
import json
from utils import setup_logger

logger = setup_logger('code_builder_agent', 'code_builder_agent.log')

class CodeBuilderAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["parsed_data"],
            template=(
                "You are an expert in generating FEniCS-based simulation code.\n\n"
                "You are given a parsed JSON object describing the problem, along with the full clarified natural language description.\n\n"
                "You must generate valid, executable Python code using FEniCS, referencing BOTH the parsed JSON and the full_text.\n\n"
                "If any important information is missing in the parsed JSON (e.g., complex geometry, fiber arrangements, custom material properties), infer and supplement it from the full_text.\n\n"
                "Always prioritize correctness, physical realism, and FEniCS best practices.\n\n"
                "Supported problem types:\n"
                "- 'fluid': Navier-Stokes\n"
                "- 'heat': heat transfer\n"
                "- 'elasticity': linear elasticity\n"
                "- 'hyperelasticity': hyperelastic materials\n"
                "- 'phase_field_fracture': nonlinear coupled system with damage variable d (range 0–1)\n\n"
                "[Code generation requirements]\n"
                "- Do NOT use Markdown formatting. Return plain Python code only.\n"
                "- Follow this structure:\n"
                "  1. Import libraries\n"
                "  2. Define geometry and mesh (either construct shapes or load .xml file)\n"
                "  3. Define function spaces (combine or separate for u/d in phase-field)\n"
                "  4. Apply boundary and initial conditions\n"
                "     - Use `expr.t = t` for time-dependent Dirichlet conditions\n"
                "     - Prefer `interpolate()` over `project()`\n"
                "     - For damage: set initial value as `interpolate(Constant(1e-4), V)`\n"
                "     - Avoid fixing damage on entire boundary; use minimal regions (e.g., bottom)\n"
                "     - Apply displacement only on relevant components: `DirichletBC(V.sub(0).sub(1), ...)`\n"
                "     - When loading .xml mesh, include boundary markers from `_facet_region.xml` if available\n"
                "  5. Define weak form F and Jacobian J; use `solve()`\n"
                "     - For time-dependent heat problems:\n"
                "       📌 Use Backward Euler by default\n"
                "       📌 Weak form: F = u*v*dx + dt*k*dot(grad(u), grad(v))*dx - u_n*v*dx\n"
                "       📌 Do NOT include grad(u_n) in any term\n"
                "  6. If problem is time-dependent, include a time-stepping loop:\n"
                "     - For phase_field_fracture: alternate solving u and d\n"
                "     - Save result each step using XDMF format\n"
                "  7. Save outputs (.xdmf files per field, with timestep info if applicable)\n\n"
                "[Code guidelines]\n"
                "- Output must be a fully executable Python script (.py), no markdown or explanation\n"
                "- Use `solve(F == 0, ...)` and include `solver_parameters` (e.g., max iterations, relaxation)\n"
                "- Prefer `MUMPS` as linear solver for nonlinear problems\n"
                "- Use `+ eps` for denominators to improve numerical stability (e.g., `epsilon + eps`)\n"
                "- When XML mesh is used:\n"
                "  1. Load with `Mesh('mesh.xml')`\n"
                "  2. If exists, load boundary markers with `MeshFunction('size_t', mesh, 'mesh_facet_region.xml')`\n"
                "  3. Use facet ids in boundary conditions\n\n"
                "Input Parameters:\n"
                "{parsed_data}"
            )
        )
        self.chain = self.prompt_template | self.llm

    def build_code(self, parsed_data: dict) -> str:
        logger.info(f"Building code from parsed data: {parsed_data}")
        data_str = json.dumps(parsed_data, ensure_ascii=False)
        try:
            response = self.chain.invoke({"parsed_data": data_str})
            code = response.content.strip()
        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            return ""
        logger.info(f"Generated code: {code}")
        return code
