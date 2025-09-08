# Fix Unicode cho Windows console
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

# Load mô hình và data
model = joblib.load('recipe_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
X = joblib.load('features_matrix.pkl')
df = pd.read_pickle('recipes_df.pkl')


# Simple map tiếng Việt → English raw ingredient
ing_map = {
    'gà': 'chicken', 'thịt bò': 'beef', 'cà rốt': 'carrot', 'hành': 'onion',
    'gạo': 'rice', 'táo': 'apple', 'sữa': 'milk', 'khoai tây': 'potato',
    'cá': 'fish', 'trứng': 'egg', 'bột mì': 'flour', 'muối': 'salt'
    # Thêm nếu cần!
}

def map_user_ingredients(ingredients):
    mapped = []
    for ing in ingredients:
        ing_lower = ing.lower().strip()
        english_ing = ing_map.get(ing_lower, ing_lower)  # Fallback giữ nguyên
        # Clean quantity nếu có (dù user ít nhập)
        clean_ing = re.sub(r'^\d+[\s\w\.,]*\s+(.*)', r'\1', english_ing).strip()
        if clean_ing and len(clean_ing) > 1:
            mapped.append(clean_ing)
    return mapped

def suggest_recipes(user_ingredients, top_k=5, use_supervised=False):
    user_raw = map_user_ingredients(user_ingredients)
    if not user_raw:
        return [], []  # Trả empty để bot handle
    user_str = ' '.join(user_raw)
    user_vec = vectorizer.transform([user_str])
    
    if use_supervised:
        pred_name = model.predict(user_vec)[0]
        recipe = df[df['title'] == pred_name].iloc[0]
        return [pred_name], [recipe]
    else:
        sim = cosine_similarity(user_vec, X).flatten()
        top_idx = np.argsort(sim)[-top_k:][::-1]
        titles = [df.iloc[idx]['title'] for idx in top_idx if sim[idx] > 0.1]  # Threshold để tránh match kém
        recipes = [df.iloc[idx] for idx in top_idx if sim[idx] > 0.1]
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
print("Hi! I am Smart Recipe Buddy offline. Type 'quit' to stop.")
while True:
    user_input = input("What ingredients do you have? (Ex: ga, ca rot, hanh): ").strip()
    if user_input.lower() == 'quit':
        print("Goodbye! Enjoy your meal <3")
        break
    ingredients = [ing.strip() for ing in user_input.split(',')]
    titles, recipes = suggest_recipes(ingredients, use_supervised=False)
    
    if titles:
        print("Here are your recipe options:")
        for i, title in enumerate(titles, 1):
            print(f"{i}. {title}")
        while True:
            try:
                while True:
                    choice = int(input("Pick a number: ").strip()) - 1  # Chuyển về index 0-based
                    details = get_recipe_details(choice, recipes)
                    print(f"\n{details}\n")
                    if "Hi! Invalid choice. Please pick a number from the list!" not in details:
                        break
                break
            except ValueError:
                print("Please enter a number only!\n")
    else:
        print("No recipes found for these ingredients. Try again!\n")
