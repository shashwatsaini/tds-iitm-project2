import base64
import os
from langchain.tools import Tool

# ---------- Read Image TOOL ----------
def read_image_tool(file_path: str) -> str:
    print(f"""
          \n-------------------------------------------------
          \n[Tool Log] ReadImageTool invoked with file_path: {file_path}
          \n-------------------------------------------------\n""")
    
    try:
        ext = os.path.splitext(file_path)[1].lstrip(".").lower()
        
        with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:image/{ext};base64,{encoded_image}"
    
    except Exception as e:
        return f"[OCR Tool Error] {str(e)}"
