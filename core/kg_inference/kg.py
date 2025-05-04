import joblib
import pandas as pd

# ============================
# 1. Load saved data
# ============================
# Load the previously saved data
data = joblib.load('/home/adarsh/Desktop/Personal Projects/NutriAI/core/kg_inference/food_matching_data.pkl')

# Extract data from the loaded object
food_nutrient_map = data['food_nutrient_map']
food_extra_map = data['food_extra_map']
kg_foods = data['kg_foods']
nutrient_foods = data['nutrient_foods']
extra_foods = data['extra_foods']
kg_embeddings = data['kg_embeddings']
nutrient_embeddings = data['nutrient_embeddings']
extra_embeddings = data['extra_embeddings']

# ============================
# 2. Load your datasets again for inference
# ============================
# Make sure you reload your original dataframes (kg_df, nutrient_data, df_nutri_value) for inference
kg_df = pd.read_csv("/home/adarsh/Desktop/Personal Projects/NutriAI/core/kg_inference/kg_data.csv")  # Example, ensure it's loaded from wherever you saved it
nutrient_data = pd.read_csv("/home/adarsh/Desktop/Personal Projects/NutriAI/core/kg_inference/nutrient_data.csv")
df_nutri_value = pd.read_csv("/home/adarsh/Desktop/Personal Projects/NutriAI/core/kg_inference/df_extra.csv")

# ============================
# 3. Clean text function for input
# ============================
def clean_text(text):
    return text.strip().lower()

# ============================
# 4. Inference function
# ============================
def full_disease_food_recommendation(disease_query, kg_df, nutrient_data, df_extra):
    disease_query = clean_text(disease_query)

    treat_df = kg_df[(kg_df['relationship'] == 'treats') & (kg_df['disease'].str.lower() == disease_query)]
    if treat_df.empty:
        return f"❌ No treatment foods found for '{disease_query.title()}'."

    output = []
    for _, row in treat_df.iterrows():
        food = row['food']
        evidence = row['evidence']
        mapped_nutrient_food = food_nutrient_map.get(food)
        mapped_extra_food = food_extra_map.get(food)

        nutrients_list = []
        extra_nutrients_list = []

        # Main nutrients (top 5 by value)
        if mapped_nutrient_food:
            nutrients = nutrient_data[nutrient_data['food'] == mapped_nutrient_food]
            nutrients_list = (
                nutrients[['nutrient', 'value']]
                .drop_duplicates(subset='nutrient', keep='first')
                .sort_values(by='value', ascending=False)
                .head(5)
                .to_dict(orient='records')
            )

        # Extra nutrient values
        if mapped_extra_food:
            extra_row = df_extra[df_extra['Main food description'] == mapped_extra_food]
            if not extra_row.empty:
                nutrient_cols = ['Energy (kcal)', 'Protein (g)', 'Carbohydrate (g)', 
                                 'Sugars, total\n(g)', 'Fiber, total dietary (g)', 'Total Fat (g)']
                for col in nutrient_cols:
                    if col in extra_row.columns:
                        val = extra_row.iloc[0][col]
                        extra_nutrients_list.append({'nutrient': col.strip(), 'value': val})

        output.append({
            "food": food.title(),
            "evidence": evidence,
            "nutrients": nutrients_list,
            "extra_nutrients": extra_nutrients_list
        })

    return output

# ============================
# 5. Example inference
# ============================
query = "asthma"  # Change the disease query here
recommendations = full_disease_food_recommendation(query, kg_df, nutrient_data, df_nutri_value)

if isinstance(recommendations, str):
    print(recommendations)
else:
    for idx, rec in enumerate(recommendations, 1):
        print(f"\n🔹 Recommendation {idx}: {rec['food']}")
        print("  ➤ Justification (evidence):", rec['evidence'])

        if rec['nutrients']:
            print("  ➤ Contains Nutrients (Top 5):")
            for nut in rec['nutrients']:
                print(f"    - {nut['nutrient'].title()} ({nut['value']})")
        else:
            print("  ➤ No main nutrient info available.")

        if rec['extra_nutrients']:
            print("  ➤ Extra Nutritional Info:")
            for extra in rec['extra_nutrients']:
                print(f"    - {extra['nutrient']} ({extra['value']})")
        else:
            print("  ➤ No extra nutrient data available.")

