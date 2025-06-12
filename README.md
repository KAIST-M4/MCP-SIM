# MCP-SIM (Memory-Coordinated Physics-Aware Simulation)
Donggeun Park<sup>a</sup>, Hyunbin Moon<sup>a</sup>, and Seunghwa Ryu<sup>a</sup><sup>*</sup>
<font size="1">이 부분의 글자 크기가 줄어듭니다.</font>
<sup>a</sup> Department of Mechanical Engineering, Korea Advanced Institute of Science and Technology (KAIST), Daejeon 34141, Republic of Korea

![License](https://img.shields.io/badge/license-MIT-green) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15645333.svg)](https://doi.org/10.5281/zenodo.15645333)
## Code Overview
Physics-based simulations are essential in science and engineering, yet creating them typically requires expert knowledge of numerical solvers and governing equations. Large language models (LLMs) offer new possibilities for natural language-based simulation, but they often fail when prompts are vague, incomplete, or multilingual. We present **MCP-SIM (Memory-Coordinated Physics-Aware Simulation), a self-correcting multi-agent framework that transforms underspecified prompts into validated simulations and explanatory reports.** The system integrates input clarification, code generation, error diagnosis, and multilingual explanation through structured agent collaboration and persistent memory. Rather than relying on one-shot code generation, MCP-SIM emulates expert-like reasoning via iterative plan–act–reflect–revise cycles (**Figure 1**). Tested on a twelve-task benchmark across diverse physics domains, MCP-SIM achieved 100% success, significantly outperforming baseline LLMs. In addition to numerical accuracy, the system produces interpretable, language-localized reports that explain each simulation’s physical logic. MCP-SIM represents a step toward general-purpose autonomous scientific assistants that simulate, adapt, and teach through natural language.
![Figure 1](https://github.com/user-attachments/assets/07fd5769-e85d-4504-b93d-69760635c7a4)

## Overview of MCP-SIM for automation of numerical simulation:
We assessed the robustness of the MCP-SIM system using a twelve-task benchmark suite that covers a variety of physics domains, including:
**Linear elasticity, Heat conduction, Fluid flow, Thermo-mechanical coupling, Piezoelectric deformation, Phase-field fracture mechanics**
For the 12 benchmark tasks, MCP-SIM demonstrates the code automation ability to generate physically meaningful results, even in the presence of ambiguity or missing information (**Figure 2**)

![Figure 3](https://github.com/user-attachments/assets/1aedd43d-bae7-4eb1-ad1e-b2c31d9aa61e)



