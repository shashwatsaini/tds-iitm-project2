from langchain.tools import Tool
import matplotlib.pyplot as plt
import pandas as pd
import os

# ---------- CODE EXECUTOR TOOL ----------
def code_executor(code: str) -> str:
    print(f"""
          \n-------------------------------------------------
          \n[Tool Log] CodeExecutorTool invoked with code: {code}
          \n-------------------------------------------------\n""")
    
    try:
        exec_globals = {}
        exec(code, exec_globals)

        # Check for saved plot
        plot_files = [f for f in os.listdir() if f.endswith('.png')]
        if plot_files:
            return f"Plot saved as {plot_files[-1]} in {os.getcwd()}"
        else:
            return "Code executed, but no PNG plot was saved."

    except Exception as e:
        return f"[CodeExecutor Error] {str(e)}"
    
code_executor_tool = Tool(
    name="CodeExecutorTool",
    func=code_executor,
    description="Executes matplotlib-based Python code to generate and save plots. Use when code must be executed dynamically."
)
