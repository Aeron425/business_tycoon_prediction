# business_tycoon_prediction

# Plan

Step 1
Extract data with OCR (pytesseract) from screenshots of the game, get about 500-1000 rows of data for first batch, a tenth or a fifth of the dataset coming from each commodity

Step 2
Analyze the data and create a set of features with the price or change in price as the label

Potental features:

- Price
- Time
- Average change in a period of time
- Highest price and lowest price in period of time
- Highest change price and lowest change in price, up and down
- More metrics probably

Step 3
Process with an ML Model

Models to Try:

Traditional Models
- Random Forest | Best Output at 64% Accuracy
- XGBoost Random Forest | Worse Output than Random Forest
- XGBoost Classifier | Worse Output than Random Forest
- Maybe some others

Neural Networks
- RNN (Recurrent Neural Network) | Not Tried
- LSTM (Long Short Term Memory) | Not Tried
- Other RNN Offshoots

# Current Agenda

Priorities
1. Get more data
2. Implement neural networks
3. Make this presentable for final project

# DO NOT OPEN ZIP_BOMB.ZIP PLEASE
