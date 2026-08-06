import os
import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

MODEL_PATH = "better-than-you/mcq-bert-model"

st.set_page_config(page_title="MCQ Predictor", page_icon="🧠", layout="centered")

st.title("Smart MCQ Solver")
st.caption("Selects the most probable option using a fine-tuned BERT model.")


@st.cache_resource
def load_mcq_model(path):
    # Fetch token from Streamlit Secrets
    hf_token = st.secrets.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(path, token=hf_token)
    model = AutoModelForMultipleChoice.from_pretrained(path, token=hf_token)

    device = torch.device("cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


with st.spinner("Loading model weights from private Hugging Face repo..."):
    tokenizer, model, device = load_mcq_model(MODEL_PATH)

prompt = st.text_area(
    "Question Prompt:",
    value="Pick the best possible answer: What is the estimated redshift of CEERS-93316?",
    height=100,
)

st.subheader("Options")
col1, col2 = st.columns(2)

with col1:
    opt_a = st.text_input("Option A:", value="Approximately z = 6.0")
    opt_b = st.text_input("Option B:", value="Approximately z = 16.7")
    opt_c = st.text_input("Option C:", value="Approximately z = 3.0")

with col2:
    opt_d = st.text_input("Option D:", value="Approximately z = 10.0")
    opt_e = st.text_input("Option E:", value="Approximately z = 13.0")

if st.button("Predict Best Answer", type="primary", use_container_width=True):
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]
    label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    if not prompt.strip() or any(not opt.strip() for opt in options):
        st.warning("Please fill out the question prompt and all 5 options.")
    else:
        with st.spinner("Calculating probabilities..."):
            prompts_repeated = [prompt] * 5
            encoding = tokenizer(
                prompts_repeated,
                options,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].unsqueeze(0).to(device)
            attention_mask = encoding["attention_mask"].unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                logits = outputs.logits.cpu().squeeze(0)
                probs = torch.softmax(logits, dim=0).numpy()

            best_idx = int(np.argmax(probs))
            best_option = label_map[best_idx]
            best_text = options[best_idx]
            confidence = probs[best_idx] * 100

            st.success("Prediction Complete!")
            st.markdown(f"### **Option {best_option}**: {best_text}")
            st.info(f"**Confidence**: {confidence:.2f}%")