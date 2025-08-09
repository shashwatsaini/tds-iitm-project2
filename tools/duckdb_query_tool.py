from langchain.tools import Tool
import duckdb

# ---------- DUCKDB QUERY EXECUTOR TOOL ----------
def duckdb_executor(sql: str) -> str:
    print(f"""
          \n-------------------------------------------------
          \n[Tool Log] DuckDBQueryTool invoked with SQL: {sql}
          \n-------------------------------------------------\n""")
    
    try:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL parquet; LOAD parquet;")
        result_df = con.execute(sql).fetchdf()
        return result_df.to_string()[:10000]
    except Exception as e:
        return f"[DuckDBQueryTool Error] {str(e)}"

duckdb_tool = Tool(
    name="DuckDBQueryTool",
    func=duckdb_executor,
    description="Useful for querying Parquet metadata from S3."
)

