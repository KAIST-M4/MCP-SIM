# simulation_executor_agent.py
import subprocess
import os
import re
from utils import setup_logger

logger = setup_logger('simulation_executor_agent', 'simulation_executor_agent.log')

class SimulationExecutorAgent:
    def __init__(self):
        self.simulation_filename = "generated_simulation.py"

    def save_code_to_file(self, code: str):
        """
        Save the generated Python simulation code to a file.
        """
        with open(self.simulation_filename, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"Simulation code saved to {self.simulation_filename}")

    def execute_simulation(self) -> str:
        """
        Executes the saved Python script and captures its output.
        If any known error patterns are found in stdout or stderr, returns the combined output as an error log.

        Returns:
            str: Simulation output log or detected error message.
        """
        try:
            result = subprocess.run(
                ["python", self.simulation_filename],
                capture_output=True,
                text=True,
                check=True
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            error_pattern = re.compile(
                r"(Traceback|NameError|SyntaxError|Exception|RuntimeError|"
                r"ImportError|ValueError|TypeError|AttributeError|IndexError|KeyError|"
                r"IndentationError|ZeroDivisionError|MemoryError|"
                r"FileNotFoundError|ModuleNotFoundError|FloatingPointError|OSError|"
                r"DijitsoError|Unable to compile|Segmentation fault|Killed|"
                r"ArityMismatch|UFLException|form compilation failed|"
                r"compute_form_data|ffc.jit|ffcjitsigning|map_expr_dag|"
                r"check_integrand_arity|check_form_arity|analyze_ufl_objects|compile_form)"
            )
            if error_pattern.search(stdout) or error_pattern.search(stderr):
                combined_error = "\n".join(filter(None, [stdout, stderr]))
                logger.error("Error detected in simulation output.")
                logger.error(f"Simulation error output: {combined_error}")
                return combined_error

            logger.info("Simulation executed successfully.")
            logger.info(f"Simulation output: {stdout}")
            return stdout

        except subprocess.CalledProcessError as e:
            logger.error("Simulation execution failed with exception.")
            logger.error(f"Error output: {e.stderr}")
            return e.stderr