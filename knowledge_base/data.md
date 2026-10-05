# Data and Methodology

## Where does the data come from?
The current release uses a synthetic dataset calibrated to approximate 2024-25 resale price levels per square foot in each locality. It is designed to behave like the real market, including location premiums, age depreciation, floor premiums and metro proximity, but individual listings are not real properties. A future release will move to licensed transaction data.

## How often is the model updated?
The model is retrained whenever the dataset is refreshed. The training date is recorded with the model metrics and shown on the Model Evaluation page.

## What are comparable properties?
Comparable properties are the listings in the dataset that are most similar to yours in the same city, matched on locality, area, bedrooms, age, property type and furnishing. They help you sanity-check the estimate against actual examples.
