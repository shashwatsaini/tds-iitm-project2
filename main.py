import os
import uuid
import base64
from datetime import datetime
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from workflow import create_workflow

app = Flask(__name__)
load_dotenv()

graph = create_workflow()

def clean_answer(ans):
    if isinstance(ans, str):
        ans = ans.strip()
        if ans.startswith("[") and ans.endswith("]"):
            ans = ans[1:-1].strip()
            if (ans.startswith('"') and ans.endswith('"')) or (ans.startswith("'") and ans.endswith("'")):
                ans = ans[1:-1]
    return ans

@app.route("/api", methods=["POST"])
def run_graph():
    try:
        if 'questions.txt' not in request.files:
            return jsonify({"error": "questions.txt is required"}), 400

        request_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex
        input_dir = os.path.join("inputs", request_id)
        output_dir = os.path.join("outputs", request_id)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Save uploaded files
        questions_file = request.files['questions.txt']
        questions_path = os.path.join(input_dir, "questions.txt")
        questions_file.save(questions_path)
        with open(questions_path, "r", encoding="utf-8") as f:
            questions_text = f.read().strip()

        image_path = None
        if 'image.png' in request.files:
            image_path = os.path.join(input_dir, "image.png")
            request.files['image.png'].save(image_path)

        csv_dir = None
        csv_text = None
        if 'data.csv' in request.files:
            csv_dir = os.path.join(input_dir, "data.csv")
            csv_text = f'If a CSV file is given, it is saved in {csv_dir}. I must only use \'CodeExecutorTool\' to interact with CSV files if a CSV file is given'
            request.files['data.csv'].save(csv_dir)

        result = graph.invoke({
            "input": questions_text,
            "request_id": request_id,
            "csv_dir": csv_dir,
            "csv_text": csv_text,
            "output_dir": output_dir
        })

        print(result['answers'])

        # Convert image paths in answers to base64
        answers_list = []
        for answer in result["answers"].values():
            if isinstance(answer, str):
                ext = os.path.splitext(answer)[1].lstrip(".").lower()
                if ext in {"png", "jpg", "jpeg"} and os.path.isfile(answer):
                    with open(answer, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                    answer = f"data:image/{ext};base64,{encoded_image}"
            answers_list.append(answer)

        answers_list = [clean_answer(ans) for ans in answers_list]        

        return jsonify(
            answers_list
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
