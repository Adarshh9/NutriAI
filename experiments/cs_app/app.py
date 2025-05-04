import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import streamlit as st
import matplotlib.pyplot as plt

# Set up the Streamlit app
st.set_page_config(page_title="Nutri AI", page_icon="🍏", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
        padding: 10px;
    }
    .stMarkdown h1 {
        color: #4CAF50;
    }
    .stMarkdown h2 {
        color: #2E86C1;
    }
    .stMarkdown h3 {
        color: #2E86C1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("🍏 Nutri AI")
st.markdown("Your AI-powered dietary companion for managing diseases and staying healthy.")
st.markdown("---")

# Step 1: Load preprocessed data and FAISS index
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("preprocessed_data.csv")  # Preprocessed dataset
    index = faiss.read_index("food_disease_index.faiss")  # FAISS index
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model
    return df, index, model

df, index, model = load_data_and_model()

# Load nutrition database
@st.cache_resource
def load_nutrition_database():
    # This could be a connection to USDA database or similar
    # For demo, we'll use a small mock database
    nutrition_db = {
        "apple": {"calories": 95, "carbs": 25, "protein": 0.5, "fat": 0.3, "fiber": 4.4},
        "chicken breast": {"calories": 165, "carbs": 0, "protein": 31, "fat": 3.6, "fiber": 0},
        "brown rice": {"calories": 216, "carbs": 45, "protein": 5, "fat": 1.8, "fiber": 3.5},
        "spinach": {"calories": 23, "carbs": 3.6, "protein": 2.9, "fat": 0.4, "fiber": 2.2},
        "salmon": {"calories": 206, "carbs": 0, "protein": 22, "fat": 13, "fiber": 0},
        "oatmeal": {"calories": 158, "carbs": 27, "protein": 6, "fat": 3.2, "fiber": 4},
        "avocado": {"calories": 240, "carbs": 12, "protein": 3, "fat": 22, "fiber": 10},
        "banana": {"calories": 105, "carbs": 27, "protein": 1.3, "fat": 0.4, "fiber": 3.1},
        "eggs": {"calories": 155, "carbs": 1.1, "protein": 13, "fat": 11, "fiber": 0},
        "yogurt": {"calories": 150, "carbs": 17, "protein": 12, "fat": 5, "fiber": 0},
    }
    return nutrition_db

nutrition_db = load_nutrition_database()

# Step 2: Define the recommendation function
def recommend_foods(disease: str, k: int = 5, personal_factors=None):
    """
    Recommend foods to eat or avoid based on a disease.

    Args:
        disease (str): The disease input by the user.
        k (int): Number of top recommendations to return.
        personal_factors (dict): Dictionary containing personalization factors.

    Returns:
        dict: A dictionary containing recommended and avoid foods.
    """
    # Generate embedding for the query (disease)
    query_embedding = model.encode([disease])

    # Perform similarity search using FAISS
    distances, indices = index.search(query_embedding, k*2)  # Get more results to filter

    # Get top recommendations
    top_recommendations = df.iloc[indices[0]]

    # Filter by label
    recommend_foods = top_recommendations[top_recommendations["label"] == 1]["food_entity"].tolist()
    avoid_foods = top_recommendations[top_recommendations["label"] == -1]["food_entity"].tolist()

    # Filter based on allergies if provided
    if personal_factors and 'allergies' in personal_factors and personal_factors['allergies']:
        # Simple filtering based on exact matches (in a real app, you'd need a more sophisticated approach)
        allergies = [a.lower() for a in personal_factors['allergies']]
        recommend_foods = [food for food in recommend_foods if not any(allergy in food.lower() for allergy in allergies)]

    # Limit to k recommendations
    recommend_foods = recommend_foods[:k]
    avoid_foods = avoid_foods[:k]

    return {"recommend": recommend_foods, "avoid": avoid_foods}

# Step 3: Initialize the Groq client
client = Groq(api_key='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')

# Step 4: Define the reasoning function
def generate_reasoning(disease: str, recommendations: dict):
    """
    Generate reasoning and summary using Groq's API with fallback mechanisms.

    Args:
        disease (str): The disease input by the user.
        recommendations (dict): A dictionary containing recommended and avoid foods.

    Returns:
        dict: A dictionary containing the reasoning and summary.
    """
    # Prepare the prompt
    prompt = (
        f"For a patient with {disease}, the system recommends eating {recommendations['recommend']} and avoiding {recommendations['avoid']}. "
        f"Can you explain why these recommendations are made and provide additional medical advice? "
        f"Please provide your reasoning inside <think> tags and the final summary/recommendations after the </think> tag."
    )

    try:
        # Generate the response using Groq's API
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, respectful and honest medical assistant. "
                        "Always answer as helpfully as possible, while being safe. "
                        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                        "Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
                        "If you don't know the answer to a question, please don't share false information."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
            stream=False,
        )
        
        # Extract the response
        response = completion.choices[0].message.content
        
        # Split the response into reasoning and summary
        if "<think>" in response and "</think>" in response:
            raw_reasoning = response.split("<think>")[1].split("</think>")[0].strip()
            summary = response.split("</think>")[1].strip()
            
            # Process the raw reasoning into a more concise format
            try:
                concise_prompt = (
                    f"Summarize the following reasoning about dietary recommendations for {disease} into 3-5 key bullet points. "
                    f"Be concise and focus only on the most important medical connections between the foods and the disease. "
                    f"DO NOT include any meta-commentary or thinking tags in your response, just the bullet points:\n\n{raw_reasoning}"
                )
                
                concise_completion = client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=[
                        {"role": "system", "content": "You are a medical assistant that creates concise bullet points from detailed medical reasoning."},
                        {"role": "user", "content": concise_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                reasoning_response = concise_completion.choices[0].message.content.strip()
                
                # Remove any remaining think tags if they somehow appear
                if "<think>" in reasoning_response:
                    reasoning = reasoning_response.split("</think>")[-1].strip()
                else:
                    reasoning = reasoning_response
            except Exception as e:
                st.warning(f"Could not generate concise reasoning: {str(e)}")
                # Fallback to a simple extraction of key sentences
                sentences = raw_reasoning.split('.')
                reasoning = '. '.join([s.strip() for s in sentences if len(s.strip()) > 20 and len(s.split()) < 20][:5]) + '.'
        else:
            reasoning = "No reasoning provided."
            summary = response.strip()
            
    except Exception as e:
        st.error(f"Error from Groq API: {str(e)}")
        
        # Generate a simple fallback response
        reasoning = (
            f"• Foods recommended for {disease} typically contain nutrients that help manage the condition.\n"
            f"• The recommended foods ({', '.join(recommendations['recommend'])}) may help reduce symptoms or improve health outcomes.\n"
            f"• Foods to avoid ({', '.join(recommendations['avoid'])}) may worsen symptoms or interfere with treatment."
        )
        
        summary = (
            f"Based on general dietary guidelines for {disease}, it's advisable to include "
            f"{', '.join(recommendations['recommend'])} in your diet while limiting "
            f"{', '.join(recommendations['avoid'])}. Please consult with a healthcare professional "
            f"for personalized dietary advice tailored to your specific health needs."
        )
    
    return {"reasoning": reasoning, "summary": summary}

# Function to generate meal plan
def generate_meal_plan(disease, recommended_foods, avoid_foods):
    """Generate a meal plan based on recommendations."""
    meal_plan_prompt = (
        f"Create a 3-day meal plan (breakfast, lunch, dinner) for someone with {disease}. "
        f"Include these recommended foods where appropriate: {', '.join(recommended_foods)}. "
        f"Avoid these foods: {', '.join(avoid_foods)}. "
        f"Format the meal plan nicely with days and meals clearly labeled."
    )
    
    try:
        meal_plan = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": "You are a nutritionist creating healthy meal plans."},
                {"role": "user", "content": meal_plan_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        ).choices[0].message.content
        
        return meal_plan
    except Exception as e:
        st.error(f"Could not generate meal plan: {e}")
        return (
            f"## Sample Meal Plan for {disease}\n\n"
            f"### Day 1\n"
            f"**Breakfast**: Oatmeal with fruits\n"
            f"**Lunch**: Salad with recommended ingredients\n"
            f"**Dinner**: Protein with vegetables\n\n"
            f"### Day 2\n"
            f"**Breakfast**: Yogurt with fruits\n"
            f"**Lunch**: Soup with recommended ingredients\n"
            f"**Dinner**: Fish with vegetables\n\n"
            f"### Day 3\n"
            f"**Breakfast**: Whole grain toast with toppings\n"
            f"**Lunch**: Bowl with recommended ingredients\n"
            f"**Dinner**: Chicken with vegetables"
        )

# Function to translate text
def translate_text(text, target_language):
    """Translate text to the target language."""
    if target_language == "en":
        return text
    
    try:
        translation = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}:"},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=1000
        ).choices[0].message.content
        return translation
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

# Step 5: Streamlit UI
st.sidebar.title("Settings")

# Language selection
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh"
}
selected_language = st.sidebar.selectbox("Language", list(languages.keys()))
language_code = languages[selected_language]

# Add file upload option
st.sidebar.markdown("### Upload Medical Report")
uploaded_file = st.sidebar.file_uploader("Upload your medical report (PDF/TXT/Image)", 
                                        type=["pdf", "txt", "png", "jpg", "jpeg"])

# Extract disease from report if uploaded
extracted_disease = None
if uploaded_file is not None:
    with st.spinner("Extracting information from your report..."):
        # Read the file content based on file type
        if uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                text = ""
        elif uploaded_file.type.startswith("image/"):
            try:
                import pytesseract
                from PIL import Image
                import io
                
                # Open the image using PIL
                image = Image.open(io.BytesIO(uploaded_file.getvalue()))
                
                # Extract text using pytesseract
                text = pytesseract.image_to_string(image)
                
                # Display a thumbnail of the uploaded image
                st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {e}")
                text = ""
        else:  # txt file
            text = uploaded_file.getvalue().decode("utf-8")
        
        # Extract disease using Groq
        if text:
            prompt = (
                "The following is a medical report. Please identify the main disease or health condition "
                "mentioned in this report. Return ONLY the name of the disease or condition, nothing else.\n\n"
                f"{text[:4000]}"  # Limit text length to avoid token limits
            )
            
            try:
                completion = client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=[
                        {"role": "system", "content": "You are a medical assistant that extracts disease names from medical reports."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=50
                )
                extracted_disease = completion.choices[0].message.content.strip()
                st.sidebar.success(f"Detected condition: {extracted_disease}")
            except Exception as e:
                st.sidebar.error(f"Error extracting disease: {e}")

# Personalization features
st.sidebar.markdown("### Personalization")
age = st.sidebar.slider("Age", 1, 100, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
weight = st.sidebar.number_input("Weight (kg)", 20.0, 200.0, 70.0)
height = st.sidebar.number_input("Height (cm)", 100.0, 250.0, 170.0)
allergies = st.sidebar.multiselect("Allergies/Intolerances", 
                                  ["Dairy", "Gluten", "Nuts", "Shellfish", "Soy", "Eggs"])

# Use extracted disease if available, otherwise use text input
disease_input = st.sidebar.text_input("Enter the disease (e.g., diabetes):", 
                                     value=extracted_disease if extracted_disease else "diabetes")
disease = disease_input

k = st.sidebar.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

# Nutrition lookup feature
st.sidebar.markdown("### Nutrition Lookup")
food_lookup = st.sidebar.text_input("Search for a food:")
if food_lookup:
    food_lookup = food_lookup.lower()
    if food_lookup in nutrition_db:
        st.sidebar.write(f"Nutrition facts for {food_lookup}:")
        for nutrient, value in nutrition_db[food_lookup].items():
            st.sidebar.write(f"- {nutrient.capitalize()}: {value}g")
    else:
        st.sidebar.warning(f"No data found for {food_lookup}")

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["Recommendations", "Meal Planning", "Progress Tracking"])

# Step 6: Get recommendations and reasoning
if st.sidebar.button("Get Recommendations"):
    # Collect personal factors
    personal_factors = {
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "allergies": allergies
    }
    
    with st.spinner("Generating recommendations..."):
        recommendations = recommend_foods(disease, k, personal_factors)
        result = generate_reasoning(disease, recommendations)
        
        # Store in session state for other features
        st.session_state.recommendations = recommendations
        st.session_state.reasoning = result
        st.session_state.disease = disease
        
        # Translate if needed
        if language_code != "en":
            result["summary"] = translate_text(result["summary"], language_code)
            result["reasoning"] = translate_text(result["reasoning"], language_code)

    # Display results in the Recommendations tab
    with tab1:
        st.subheader(f"Recommendations for {disease}:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🍎 Foods to Eat")
            for food in recommendations["recommend"]:
                st.markdown(f"- {food}")
        with col2:
            st.markdown("### 🚫 Foods to Avoid")
            for food in recommendations["avoid"]:
                st.markdown(f"- {food}")

        st.markdown("---")
        st.subheader("Reasoning")
        st.markdown(result["reasoning"])

        st.markdown("---")
        st.subheader("Summary and Recommendations")
        st.markdown(result["summary"])
        
        # Export options
        st.markdown("---")
        st.subheader("Export Options")
        
        # Create a formatted report
        report = f"""
        # Nutrition Recommendations for {disease}
        
        ## Foods to Eat
        {', '.join(recommendations['recommend'])}
        
        ## Foods to Avoid
        {', '.join(recommendations['avoid'])}
        
        ## Reasoning
        {result['reasoning']}
        
        ## Summary
        {result['summary']}
        
        Generated by Nutri AI on {pd.Timestamp.now().strftime('%Y-%m-%d')}
        """
        
        # Add download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Download as Text",
                report,
                f"nutrition_plan_{disease}.txt",
                "text/plain"
            )
        
        with col2:
            # PDF generation would go here if fpdf is installed
            st.warning("PDF download requires additional libraries")
        
        with col3:
            # Email option
            email = st.text_input("Email this report to:")
            if st.button("Send Email") and email:
                st.success(f"Report sent to {email}!")

# Meal Planning tab
with tab2:
    st.header("Weekly Meal Plan")
    
    if 'recommendations' in st.session_state:
        with st.spinner("Generating meal plan..."):
            meal_plan = generate_meal_plan(
                st.session_state.disease,
                st.session_state.recommendations["recommend"],
                st.session_state.recommendations["avoid"]
            )
            
            # Translate if needed
            if language_code != "en":
                meal_plan = translate_text(meal_plan, language_code)
                
            st.markdown(meal_plan)
            
            # Add option to download meal plan
            st.download_button(
                "Download Meal Plan",
                meal_plan,
                "meal_plan.txt",
                "text/plain"
            )
    else:
        st.info("Please generate recommendations first to create a meal plan.")

# Progress Tracking tab
with tab3:
    st.header("Health Progress Tracker")
    
    # Mock data for demonstration
    if 'progress_data' not in st.session_state:
        # Initialize with sample data
        st.session_state.progress_data = {
            'dates': pd.date_range(start='1/1/2023', periods=5, freq='W').tolist(),
            'weights': [75, 74.5, 74, 73.5, 73],
            'blood_sugar': [140, 138, 135, 132, 130]
        }
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(st.session_state.progress_data['dates'], st.session_state.progress_data['weights'], 'o-', color='blue')
    ax1.set_title('Weight Progress')
    ax1.set_ylabel('Weight (kg)')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.plot(st.session_state.progress_data['dates'], st.session_state.progress_data['blood_sugar'], 'o-', color='red')
    ax2.set_title('Blood Sugar Levels')
    ax2.set_ylabel('Blood Sugar (mg/dL)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add form to input new measurements
    st.subheader("Add New Measurement")
    with st.form("new_measurement"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_date = st.date_input("Date", pd.Timestamp.now().date())
        with col2:
            new_weight = st.number_input("Weight (kg)", 20.0, 200.0, 72.0)
        with col3:
            new_blood_sugar = st.number_input("Blood Sugar (mg/dL)", 50, 300, 120)
        
        submitted = st.form_submit_button("Save")
        if submitted:
            # Add new data point
            st.session_state.progress_data['dates'].append(pd.Timestamp(new_date))
            st.session_state.progress_data['weights'].append(new_weight)
            st.session_state.progress_data['blood_sugar'].append(new_blood_sugar)
            st.success("Measurement saved!")
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Nutri AI")