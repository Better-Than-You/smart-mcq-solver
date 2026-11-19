# Smart MCQ Solver

Smart MCQ Solver is a Streamlit application that predicts the best answer to a
five-option multiple-choice question. It uses a fine-tuned BERT model hosted on
Hugging Face and displays both the selected option and the model's confidence.

## Features

- Accepts a question prompt and five answer choices (A-E).
- Runs inference with a Hugging Face `AutoModelForMultipleChoice` model.
- Uses CPU inference by default, so the app can run locally without a GPU.
- Reports the predicted option, answer text, and confidence percentage.
- Includes training experiments, milestone notebooks, and the source datasets.

## Demo

The app is built with [Streamlit](https://streamlit.io/). Once it is running,
open the local URL shown in the terminal, usually
[`http://localhost:8501`](http://localhost:8501).

## Requirements

- Python 3.9 or newer
- Access to the Hugging Face model
- A Hugging Face token when the model repository is private

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/Better-Than-You/smart-mcq-solver.git
cd smart-mcq-solver

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure Hugging Face access

The application loads the model configured in `app.py`:

```text
better-than-you/mcq-bert-model
```

If the repository is private, create `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_your_token_here"
```

Do not commit this file or expose the token in source control. If the model is
public, the token can be omitted.

## Run the application

```bash
streamlit run app.py
```

Enter a prompt and one answer for each of options A through E, then select
**Predict Best Answer**. The model scores all five choices and returns the
highest-probability option.

## Project structure

```text
.
├── app.py                    # Streamlit inference application
├── requirements.txt          # Python dependencies
├── data/
│   ├── train.csv             # Labeled training examples
│   ├── test.csv              # Unlabeled evaluation examples
│   └── sample_submission.csv # Example prediction format
├── code/                     # Model training and evaluation notebooks
└── milestones/               # Project milestone notebooks
```

## Dataset format

The training data contains the following columns:

| Column | Description |
| --- | --- |
| `id` | Example identifier |
| `prompt` | Multiple-choice question |
| `A`-`E` | Answer choices |
| `answer` | Correct option label for training |

`data/test.csv` has the same question and option columns without the `answer`
column. `data/sample_submission.csv` shows the expected prediction format for
dataset submissions.

## How inference works

1. The prompt is repeated once for each answer choice.
2. The tokenizer encodes each prompt-choice pair with truncation and padding.
3. The multiple-choice model produces one logit per option.
4. A softmax converts logits to probabilities.
5. The option with the highest probability is shown as the prediction.

## Development

Training and experimentation are documented in the Jupyter notebooks under
`code/` and `milestones/`. To work with them, install Jupyter in the virtual
environment and launch:

```bash
pip install jupyter
jupyter notebook
```

When contributing, keep credentials out of the repository, preserve the
five-option input format, and update this README when setup or usage changes.

## License

No license file is currently included in this repository. Add a license before
redistributing the project or its model and dataset artifacts.
