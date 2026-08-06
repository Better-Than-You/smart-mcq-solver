from huggingface_hub import HfApi, login

# 1. Authenticate
HF_TOKEN = "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"  # Replace with your token
login(token=HF_TOKEN)

api = HfApi()

USER_NAME = api.whoami()["name"]
MODEL_REPO_ID = f"{USER_NAME}/mcq-bert-model"
SPACE_REPO_ID = f"{USER_NAME}/dl-24f3004148-dl-project-model"
index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCQ Predictor</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            max-width: 700px;
            margin: 40px auto;
            padding: 0 20px;
            background-color: #0b0f19;
            color: #e5e7eb;
        }}
        h1 {{ color: #ffffff; text-align: center; margin-bottom: 8px; }}
        p.sub {{ text-align: center; color: #9ca3af; margin-bottom: 24px; }}
        label {{ font-weight: 600; display: block; margin-top: 12px; color: #d1d5db; }}
        textarea, input {{
            width: 100%;
            padding: 10px;
            margin-top: 6px;
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 6px;
            color: #ffffff;
            box-sizing: border-box;
        }}
        button {{
            width: 100%;
            margin-top: 20px;
            padding: 12px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }}
        button:hover {{ background: #1d4ed8; }}
        #result {{
            margin-top: 24px;
            padding: 16px;
            background: #111827;
            border-left: 4px solid #2563eb;
            border-radius: 6px;
            display: none;
        }}
    </style>
</head>
<body>
    <h1>MCQ Predictor</h1>
    <p class="sub">Selects the most probable answer using fine-tuned BERT models</p>

    <label for="prompt">Question Prompt:</label>
    <textarea id="prompt" rows="3">Pick the best possible answer: What is the estimated redshift of CEERS-93316?</textarea>

    <label for="optA">Option A:</label>
    <input type="text" id="optA" value="Approximately z = 6.0">

    <label for="optB">Option B:</label>
    <input type="text" id="optB" value="Approximately z = 16.7">

    <label for="optC">Option C:</label>
    <input type="text" id="optC" value="Approximately z = 3.0">

    <label for="optD">Option D:</label>
    <input type="text" id="optD" value="Approximately z = 10.0">

    <label for="optE">Option E:</label>
    <input type="text" id="optE" value="Approximately z = 13.0">

    <button onclick="predictAnswer()">Predict Best Answer</button>

    <div id="result"></div>

    <script>
        async function predictAnswer() {{
            const prompt = document.getElementById('prompt').value;
            const options = [
                document.getElementById('optA').value,
                document.getElementById('optB').value,
                document.getElementById('optC').value,
                document.getElementById('optD').value,
                document.getElementById('optE').value
            ];
            const labels = ['A', 'B', 'C', 'D', 'E'];
            const resultDiv = document.getElementById('result');

            resultDiv.style.display = 'block';
            resultDiv.innerHTML = "<em>Calculating probabilities via Hugging Face Inference API...</em>";

            try {{
                // Prepare pairs for Multiple Choice inference
                const inputs = options.map(opt => prompt + " " + opt);

                const response = await fetch("https://api-inference.huggingface.co/models/{MODEL_REPO_ID}", {{
                    method: "POST",
                    headers: {{
                        "Authorization": "Bearer {HF_READ_TOKEN}",
                        "Content-Type": "application/json"
                    }},
                    body: JSON.stringify({{ inputs: inputs }})
                }});

                const data = await response.json();

                if (data.error) {{
                    resultDiv.innerHTML = "<strong style='color: #ef4444;'>API Status:</strong> " + data.error + "<br><small>If model is loading, wait 30 seconds and try again.</small>";
                    return;
                }}

                // Calculate argmax from output logits/scores
                let scores = Array.isArray(data) ? (data[0].score ? data.map(d => d.score) : data) : data;
                let maxIdx = 0;
                for (let i = 1; i < scores.length; i++) {{
                    if (scores[i] > scores[maxIdx]) maxIdx = i;
                }}

                resultDiv.innerHTML = "<h3>Top Prediction:</h3>" +
                    "<p style='font-size: 18px;'><strong>Option " + labels[maxIdx] + ":</strong> " + options[maxIdx] + "</p>";

            }} catch (err) {{
                resultDiv.innerHTML = "<strong style='color: #ef4444;'>Error:</strong> " + err.message;
            }}
        }}
    </script>
</body>
</html>
"""

# 3. Space Metadata (Static SDK is 100% Free)
readme_content = """---
title: Smart MCQ Solver
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---
"""

# 4. Upload files
api.upload_file(
    path_or_fileobj=readme_content.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=SPACE_REPO_ID,
    repo_type="space",
)

api.upload_file(
    path_or_fileobj=index_html.encode("utf-8"),
    path_in_repo="index.html",
    repo_id=SPACE_REPO_ID,
    repo_type="space",
)

# Delete server-side app files if present
for old_file in ["app.py", "requirements.txt"]:
    try:
        api.delete_file(
            path_in_repo=old_file, repo_id=SPACE_REPO_ID, repo_type="space"
        )
    except Exception:
        pass

print(
    f"Deployment complete! Free Static Space live at: https://huggingface.co/spaces/{SPACE_REPO_ID}"
)
