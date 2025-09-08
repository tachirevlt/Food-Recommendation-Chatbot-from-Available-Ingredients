# Fix Unicode cho Windows console
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load mô hình và data
model = joblib.load('E:/CodeFolder/Github_project/Offline_Cooking_Chatbot/recipe_model.pkl')
vectorizer = joblib.load('E:/CodeFolder/Github_project/Offline_Cooking_Chatbot/vectorizer.pkl')
X = joblib.load('E:/CodeFolder/Github_project/Offline_Cooking_Chatbot/features_matrix.pkl')
df = pd.read_pickle('E:/CodeFolder/Github_project/Offline_Cooking_Chatbot/recipes_df.pkl')

# Simple mapping NER
ner_map = {
    'chicken': 'PROTEIN', 'beef': 'PROTEIN', 'carrot': 'VEGETABLE', 'onion': 'VEGETABLE',
    'gà': 'PROTEIN', 'thịt bò': 'PROTEIN', 'cà rốt': 'VEGETABLE', 'hành': 'VEGETABLE',
    'rice': 'GRAIN', 'gạo': 'GRAIN', 'apple': 'FRUIT', 'táo': 'FRUIT'
}
def map_to_ner(ingredients):
    mapped = []
    for ing in ingredients:
        ing_lower = ing.lower().strip()
        tag = ner_map.get(ing_lower, ing_lower.upper())  # Fallback: uppercase
        mapped.append(tag)
    return mapped

def suggest_recipes(user_ingredients, top_k=3, use_supervised=False):
    # Map user input thành NER
    user_ner = map_to_ner(user_ingredients)
    if not user_ner:
        return "Hi! I don't understand these ingredients. Try English or details like 'chicken, carrot'."
    
    user_str = ' '.join(user_ner)
    user_vec = vectorizer.transform([user_str])
    
    if use_supervised:
        pred_name = model.predict(user_vec)[0]
        recipe = df[df['title'] == pred_name].iloc[0]
        # Chỉ trả tên món, lưu recipe để chọn sau
        return [pred_name], [recipe]
    else:
        sim = cosine_similarity(user_vec, X).flatten()
        top_idx = np.argsort(sim)[-top_k:][::-1]
        titles = [df.iloc[idx]['title'] for idx in top_idx]
        recipes = [df.iloc[idx] for idx in top_idx]
        return titles, recipes

def get_recipe_details(selected_idx, recipes):
    if 0 <= selected_idx < len(recipes):
        recipe = recipes[selected_idx]
        return (f"Selected: {recipe['title']}\n"
                f"Ingredients: {recipe['ingredients']}...\n"
                f"Directions: {recipe['directions']}...\n"
                f"Link: {recipe.get('link', 'No link')}\n"
                f"Source: {recipe.get('source', 'Unknown')}")
    return "Hi! Invalid choice. Please pick a number from the list!"

# Bot loop with selection
print("Hi onii-chan! I am Smart Recipe Buddy offline. Type 'quit' to stop.")
while True:
    user_input = input("What ingredients do you have? (Ex: ga, ca rot, hanh): ").strip()
    if user_input.lower() == 'quit':
        print("Goodbye! Enjoy your meal :)")
        break
    ingredients = [ing.strip() for ing in user_input.split(',')]
    titles, recipes = suggest_recipes(ingredients, use_supervised=False)
    
    if titles:
        print("Here are your recipe options:")
        for i, title in enumerate(titles, 1):
            print(f"{i}. {title}")
        try:
            choice = int(input("Pick a number (1-3): ").strip()) - 1  # Chuyển về index 0-based
            details = get_recipe_details(choice, recipes)
            print(f"\n{details}\n")
        except ValueError:
            print("Hi! Please enter a number only!\n")
    else:
        print("No recipes found for these ingredients. Try again!\n")