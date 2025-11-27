# NutriLens: End-to-End Calorie Estimation From a Single Food Image

This repository contains the full Google Colab + PyTorch pipeline used for a **NutriLens**-style calorie estimation project.

The system takes a **single RGB food image** and predicts:

- **Per-item segmentation** (FoodSeg103 + Mask R-CNN)
- **Per-item food class** (ConvNeXt-Large on FoodSeg103 segments)
- **Dish-level mass (grams)** (Nutrition5k + depth + geometry + visual features)
- **Dish-level calories (kcal)** and macros (fat / carbs / protein) via gradient-boosted ensembles

Everything is designed to run **inside Google Colab** with all data stored in **Google Drive** under:

```text
/content/drive/MyDrive/Cal_Estimation_Project/
