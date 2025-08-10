import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import contextlib
from langchain.tools import Tool

con = duckdb.connect(database=':memory:', read_only=False)
con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")

def code_executor(code: str) -> str:
    print(f"""
          \n-------------------------------------------------
          \n[Tool Log] CodeExecutor invoked with input: {code}
          \n-------------------------------------------------\n""")
    
    try:
        global con

        code = code.strip()

        exec_globals = {"con": con, "duckdb": duckdb, "pd": pd, "plt": plt}

        # Case 1: Explicit SQL markdown block
        if code.startswith("```sql"):
            code = code.strip("` \n")[3:]
            df = con.execute(code).df()
            return df.to_string(index=False) if not df.empty else "Query executed successfully, but returned no rows."

        # Case 2: Try as Python
        if code.startswith("```python"):
            code = code.strip("` \n")[6:]
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, exec_globals, exec_globals)
            printed_output = stdout_buffer.getvalue().strip()

            if printed_output:
                return printed_output
            return "Code executed successfully, but no output was generated."
        
        # Case 3: Try as Python without markdown
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, exec_globals, exec_globals)
            printed_output = stdout_buffer.getvalue().strip()

            if printed_output:
                return printed_output
            return "Code executed successfully, but no output was generated."
        
    except Exception as e:
        return f"[CodeExecutor Error] {str(e)}"

code_executor_tool = Tool(
    name="CodeExecutorTool",
    func=code_executor,
    description=(
        "Executes Python code with pandas/matplotlib and has direct access to an active DuckDB connection "
        "via `con` or `duckdb`. Supports S3/httpfs and Parquet queries directly without passing results as tokens."
        "Executes matplotlib / seaborn code to generate plots, which are saved to the output directory."
    )
)
