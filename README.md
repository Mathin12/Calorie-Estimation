# Enhancing Calorie Estimation Accuracy through AI-Based Portion Size Prediction Using FoodSeg103 and Nutrition5k Datasets

This repository contains the full Google Colab + PyTorch pipeline used for our calorie estimation project.

The system takes a **single RGB food image** and predicts:

- **Per-item segmentation** (FoodSeg103 + Mask R-CNN)
- **Per-item food class** (ConvNeXt-Large on FoodSeg103 segments)
- **Dish-level mass (grams)** (Nutrition5k + depth + geometry + visual features)
- **Dish-level calories (kcal)** and macros (fat / carbs / protein) via gradient-boosted ensembles

Everything is designed to run **inside Google Colab** with all data stored in **Google Drive** under:

```text
/content/drive/MyDrive/Cal_Estimation_Project/
