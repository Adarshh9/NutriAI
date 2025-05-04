import streamlit as st
import pandas as pd
from groq import Groq
import pickle

# Load the trained model
with open("rf_model.pkl", "rb") as f:
    clf = pickle.load(f)
# Load the combined training data
combined_train_data = pd.read_csv("combined_train_data.csv")

# Initialize Groq client
client = Groq(api_key='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')

# Streamlit UI
st.title("🍏 NutriMumbai AI")
st.markdown("Your AI-powered dietary companion for managing diseases and staying healthy in Mumbai.")
st.markdown("---")

# Dropdown for disease selection
diseases = combined_train_data["term2"].unique().tolist()
selected_disease = st.selectbox("Select a disease:", diseases)

# Get recommendations
if st.button("Get Recommendations"):
    # Get all food-disease pairs for the selected disease
    food_disease_pairs = combined_train_data[combined_train_data["term2"] == selected_disease]

    # Select features for inference
    inference_features = food_disease_pairs[[
        "bert_cause_cs_pairs", "bert_treat_cs_pairs",
        "roberta_cause_cs_pairs", "roberta_treat_cs_pairs",
        "biobert_cause_cs_pairs", "biobert_treat_cs_pairs"
    ]]

    # Predict relationships
    predictions = clf.predict(inference_features)

    # Add predictions to the DataFrame
    food_disease_pairs["prediction"] = predictions

    # Generate recommendations (unique foods only)
    recommend_foods = list(set(food_disease_pairs[food_disease_pairs["prediction"] == 0]["term1"].tolist()))
    avoid_foods = list(set(food_disease_pairs[food_disease_pairs["prediction"] == 1]["term1"].tolist()))

    # Display recommendations
    st.subheader(f"Recommendations for {selected_disease}:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🍎 Foods to Eat")
        for food in recommend_foods:
            st.markdown(f"- {food}")
    with col2:
        st.markdown("### 🚫 Foods to Avoid")
        for food in avoid_foods:
            st.markdown(f"- {food}")

    # Generate reasoning and summary using Groq's API
    def generate_reasoning(disease: str, recommend_foods: list, avoid_foods: list):
        """
        Generate reasoning and summary using Groq's API.

        Args:
            disease (str): The selected disease.
            recommend_foods (list): List of foods to eat.
            avoid_foods (list): List of foods to avoid.

        Returns:
            dict: A dictionary containing the reasoning and summary.
        """
        # Prepare the prompt
        prompt = (
            f"For a patient with {disease}, the system recommends eating {recommend_foods} and avoiding {avoid_foods}. "
            f"Can you explain why these recommendations are made and provide additional medical advice? "
            f"Please provide your reasoning inside <think> tags and the final summary/recommendations after the </think> tag."
        )

        # Generate the response using Groq's API
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",  # Use the desired model
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, respectful and honest medical assistant. "
                        "Always answer as helpfully as possible, while being safe. "
                        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                        "Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                        "If you don’t know the answer to a question, please don’t share false information."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,  # Control the randomness of the output
            max_tokens=4096,  # Maximum number of tokens to generate
            top_p=0.95,  # Nucleus sampling parameter
            stream=False,  # Set to False for a single response
            stop=None,  # No specific stop tokens
        )

        # Extract the response
        response = completion.choices[0].message.content

        # Split the response into reasoning and summary
        if "<think>" in response and "</think>" in response:
            reasoning = response.split("<think>")[1].split("</think>")[0].strip()
            summary = response.split("</think>")[1].strip()
        else:
            reasoning = "No reasoning provided."
            summary = response.strip()

        return {"reasoning": reasoning, "summary": summary}

    # Generate reasoning and summary
    result = generate_reasoning(selected_disease, recommend_foods, avoid_foods)

    # Display reasoning and summary
    st.markdown("---")
    st.subheader("Reasoning")
    st.markdown(result["reasoning"])

    st.markdown("---")
    st.subheader("Summary and Recommendations")
    st.markdown(result["summary"])

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by NutriMumbai AI")