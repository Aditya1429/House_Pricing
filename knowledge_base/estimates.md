# How Estimates Work

## How is the price estimate calculated?
PropertyLens uses a LightGBM machine learning model trained on 6,000 residential listings across Bangalore, Mumbai and Pune. It considers locality, area, bedrooms, bathrooms, property type, age, floor, parking, furnishing, amenities and distance to the nearest metro or suburban station. The model learns the logarithm of price, which keeps errors proportional across cheap and expensive markets.

## How accurate are the estimates?
On held-out test data the typical error (MAPE) is around 8%, and R² is about 0.97. Every estimate comes with a likely range that is calibrated so roughly 80% of real prices fall inside it. Accuracy is lower for rare property types such as villas in Mumbai, and for very large or very old homes, because there are fewer examples of them. The Model Evaluation page shows full metrics, including error by city.

## Why is my estimate different from the asking price?
Asking prices are often 5 to 15% above final transaction prices, and they reflect things the model cannot see: interior condition, exact view, society reputation, builder brand, legal status and seller urgency. Treat the estimate as an independent reference point for negotiation, not a valuation certificate.
Keywords: asking price listed price seller quote different higher lower negotiation

## What do the price drivers mean?
Each estimate lists the factors that moved the price most compared with an average listing in the dataset. For example, "Location +94%" means the locality alone roughly doubles the price compared with the average listing, and "Property age -12%" means the age reduces it by about 12%. These come from SHAP contributions computed by the model.

## Is this a certified valuation?
No. PropertyLens estimates are for information and research only. For loans, legal or tax purposes you need a valuation from a bank-empanelled or government-registered valuer.

## Which cities and localities are supported?
Bangalore: Indiranagar, Koramangala, Jayanagar, HSR Layout, Hebbal, Bellandur, Whitefield, Marathahalli, Yelahanka, Electronic City. Mumbai: Worli, Bandra West, Juhu, Powai, Andheri West, Chembur, Goregaon, Borivali, Thane West, Kharghar. Pune: Koregaon Park, Kalyani Nagar, Aundh, Viman Nagar, Baner, Kharadi, Wakad, Hinjewadi, Hadapsar, Wagholi. More cities are planned.
Keywords: supported cities coverage available Delhi Chennai Hyderabad Kolkata Noida Gurgaon other cities which areas covered
